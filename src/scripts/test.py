import torch
import glob
import os
from src.config_files.config import *
from src.recommender_model import MLP
from src.commons.data_utils import load_data_queries, preprocess_data_queries
from src.commons.train_utils import test_loop
from src.config_files.logging_config import train_logger as logging

def main():
    df_songs, df_queries = load_data_queries()
    df_queries = preprocess_data_queries(df_queries)

    df = df_queries.sample(frac=1, random_state=42).reset_index(drop=True)
    train_sample = df.sample(frac=TRAIN_SPLIT, random_state=42).reset_index(drop=True)
    remaining = df.drop(train_sample.index).reset_index(drop=True)
    val_sample = remaining.sample(frac=VAL_SPLIT / (1 - TRAIN_SPLIT), random_state=42).reset_index(drop=True)
    test_sample = remaining.drop(val_sample.index).reset_index(drop=True)

    list_of_files = glob.glob('data/models/model_*.pkl')
    MODEL_PATH = max(list_of_files, key=os.path.getctime)
    
    output_size = df['label_encoded'].nunique()
    model = MLP(INPUT_SIZE, HIDDEN_SIZE, output_size).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    logging.info(f"Model loaded from {MODEL_PATH}")

    test_metrics = test_loop(test_sample=test_sample, model=model, device=DEVICE)
    logging.info(f"Test Metrics: {test_metrics}")

if __name__ == "__main__":
    main()