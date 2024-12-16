import logging
from datetime import datetime
import os

os.makedirs('logs', exist_ok=True)

log_filename_train = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
train_logger = logging.getLogger('train')
train_logger.setLevel(logging.INFO)

train_handler = logging.FileHandler(log_filename_train)
train_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
train_logger.addHandler(train_handler)