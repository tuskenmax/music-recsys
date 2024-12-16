import pandas as pd
import torch
from torch.utils.data import Dataset
from src.config_files.config import INPUT_PATH, MODEL_SAVE_PATH, DEVICE, MODEL_LOAD_PATH, LLM_MODEL
import glob
import os
from src.recommender_model import MLP


def load_data_queries():
    df_queries = pd.read_csv(f'./{INPUT_PATH}/final_train_data_music_queries.csv')
    return df_queries


def load_data_songs():
    df_songs = pd.read_csv(f'./{INPUT_PATH}/df_songs.csv')
    return df_songs


def preprocess_data_songs():
    df_songs = pd.read_csv(f'./{INPUT_PATH}/278k_song_labelled.csv')
    songs = pd.read_csv(f'./{INPUT_PATH}/TEST_labeled_songs_with_name.csv')
    top_artists = pd.read_csv(f'./{INPUT_PATH}/top10k-spotify-artist-metadata.csv')
    user_item = pd.read_csv(f"./{INPUT_PATH}/jams.csv", delimiter='\t', on_bad_lines='skip')
    # filter out top k spotify artists and logically good metrics
    K = 12000
    songs = songs[songs['instrumentalness'] < 0.5]
    songs = songs[songs['speechiness'] < 0.33]
    songs = songs[songs['tempo'] < 130]
    songs = songs[songs['liveness'] < 0.8]

    sorted_by10k = user_item[user_item['artist'].isin(top_artists['artist'][:K])]
    songs = songs[songs['uri'].isin(sorted_by10k['spotify_uri'])]

    # filter
    df_songs = df_songs[df_songs.index.isin(songs.index)]
    df_songs['url'] = songs['url']
    df_songs = df_songs.drop(columns = ['Unnamed: 0', 'duration (ms)', 'liveness', 'spec_rate', 'labels', 'instrumentalness', 'energy'])
    df_songs['tempo'] = (df_songs['tempo'] - df_songs['tempo'].min()) / (df_songs['tempo'].max() - df_songs['tempo'].min())
    df_songs['low valence'] = 1 - df_songs['valence']

    min_loudness = df_songs['loudness'].min()
    max_loudness = df_songs['loudness'].max()
    min_value = min(0, min_loudness)
    max_value = max(0, max_loudness)
    df_songs['loudness'] = (df_songs['loudness'] - min_value) / (max_value - min_value)
    df_songs.reset_index(inplace=True, drop=True)

    return df_songs


def encode_sentence(sentence):
    encoded_sentence = torch.from_numpy(LLM_MODEL.encode(sentence))
    return encoded_sentence


def preprocess_data_queries(df):
    df['label_encoded'] = pd.factorize(df['label'])[0]
    df['label_encoded'].unique()
    llm_model = LLM_MODEL
    df['sentence_embeddings'] = df['query'].apply(lambda sentence: encode_sentence(sentence))
    return df


def load_preprocessed_queries():
    df = pd.read_pickle(f'./{INPUT_PATH}/df_queries.pkl')
    return df



def load_latest_model():
    model_files = glob.glob(os.path.join(MODEL_LOAD_PATH, "*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_LOAD_PATH}")
    latest_model_path = max(model_files, key=os.path.getctime)
    model = torch.load(latest_model_path, map_location=DEVICE)
    print(f"Loaded latest model from: {latest_model_path}")
    return model


class CustomDataset(Dataset):

    def __init__(self, queries_df, songs_df):
        self.queries = queries_df['sentence_embeddings']
        self.labels = queries_df['label']
        self.songs_df = songs_df

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_embedding = self.queries[idx].clone().detach().float()
        label_column = self.labels[idx]
        label_values = torch.from_numpy(self.songs_df[label_column].values).float()
        return query_embedding, label_values