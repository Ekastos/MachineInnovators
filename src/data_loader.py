# src/data_loader.py

from datasets import load_dataset
from src import config

def load_sentiment_dataset():
    """
    Carica il dataset per l'analisi del sentiment da Hugging Face.
    """
    print(f"Caricamento del dataset '{config.DATASET_NAME}/{config.DATASET_SUBSET}'...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET)
    print("Caricamento completato.")
    return dataset