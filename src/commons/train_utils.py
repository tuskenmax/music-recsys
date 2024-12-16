import pandas as pd
import torch
import logging
from src.config_files.logging_config import train_logger 
from torch.utils.data import DataLoader
from src.commons.data_utils import CustomDataset
from src.commons.data_utils import load_data_songs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_loop(train_sample, model, optimizer, criterion, batch_size=512):
    running_loss = 0.0
    model.train()
    train_ds = CustomDataset(train_sample, load_data_songs())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    average_loss = running_loss / len(train_loader)
    train_logger.info(f"Training Loss: {average_loss:.4f}")
    return average_loss

def validation_loop(val, model, criterion, batch_size=512):
    running_loss = 0.0
    model.eval()
    val_ds = CustomDataset(val, load_data_songs())
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    average_loss = running_loss / len(val_loader)
    train_logger.info(f"Validation Loss: {average_loss:.4f}")
    return average_loss


def train(model, train_sample, val_sample, criterion, optimizer, device, epochs):
    scores = []
    losses = []
    model.to(device)
    for _ in range(epochs):
        epoch_losses = train_loop(train_sample, model, optimizer, criterion)
        epoch_scores = validation_loop(val_sample, model, criterion)

        scores.append(epoch_scores)
        losses.append(epoch_losses)
    return model



def test_loop(test_sample, model, batch_size=512, device="cuda"):
    model.eval()
    test_ds = CustomDataset(test_sample)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1 Score: {f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}