import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer

INPUT_PATH = 'data/input_data'

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

INPUT_SIZE = 384
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
OUTPUT_SIZE = None

BATCH_SIZE = 512
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LLM_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

MODEL_SAVE_PATH = f"data/models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
MODEL_LOAD_PATH = "data/models"