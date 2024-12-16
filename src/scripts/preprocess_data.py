from src.commons.data_utils import load_data_queries, preprocess_data_queries, preprocess_data_songs
from src.config_files.config import INPUT_PATH

df_queries = load_data_queries()
df_queries = preprocess_data_queries(df_queries)
df_queries.to_pickle(f'./{INPUT_PATH}/df_queries.pkl')
df_songs = preprocess_data_songs()
df_songs.to_csv(f'./{INPUT_PATH}/df_songs.csv', index=False)