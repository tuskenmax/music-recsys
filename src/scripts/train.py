import torch
import torch.nn as nn
import torch.optim as optim
from src.config_files.config import *
from src.commons.data_utils import preprocess_data_songs, load_preprocessed_queries, load_data_songs
from src.commons.train_utils import train
from src.config_files.logging_config import train_logger
from src.recommender_model import MLP
import pandas as pd


def main():

    df_queries = load_preprocessed_queries()
    df_songs = load_data_songs()
    df = df_queries.sample(frac=1, random_state=42).reset_index(drop=True)
    train_sample = df.sample(frac=TRAIN_SPLIT, random_state=42).reset_index(drop=True)
    remaining = df.drop(train_sample.index).reset_index(drop=True)
    val_sample = remaining.sample(frac=VAL_SPLIT / (1 - TRAIN_SPLIT), random_state=42).reset_index(drop=True)

    global OUTPUT_SIZE
    OUTPUT_SIZE = df_songs.shape[0]

    model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_logger.info(f"Starting training with {EPOCHS} epochs, device: {DEVICE}")

    model = train(
        model=model, 
        train_sample=train_sample, 
        val_sample=val_sample, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=DEVICE, 
        epochs=EPOCHS
    )

    torch.save(model, MODEL_SAVE_PATH)
    train_logger.info(f"Model saved to {MODEL_SAVE_PATH}")
    train_logger.info("Training completed.")

if __name__ == "__main__":
    main()