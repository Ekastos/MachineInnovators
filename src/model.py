# src/model.py

from transformers import pipeline
from src import config


def load_sentiment_pipeline(device):
    """
    Carica la pipeline di sentiment analysis pre-addestrata sul dispositivo specificato.

    Args:
        device (str): Il dispositivo su cui caricare il modello ('mps', 'cpu', ecc.).
    """
    print(f"Caricamento del modello '{config.MODEL_NAME}' sul dispositivo '{device}'...")

    # Passiamo il dispositivo alla pipeline.
    # Transformers gestirà lo spostamento del modello sulla GPU per noi.
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=config.MODEL_NAME,
        device=device
    )

    print("Modello caricato con successo.")
    return sentiment_pipeline