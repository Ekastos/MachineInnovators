# src/model.py

from transformers import pipeline
from src import config


def load_sentiment_pipeline(device, model_name=None):
    """
    Carica la pipeline di sentiment analysis.

    Args:
        device (str): Il dispositivo su cui caricare il modello ('mps', 'cpu', 'cuda').
        model_name (str, optional): Il percorso o nome del modello da caricare.
                                    Se None, usa il modello base definito in config.
    """
    # Se non viene specificato un modello, usiamo quello di default (base)
    target_model = model_name if model_name else config.MODEL_NAME

    print(f"Caricamento del modello: '{target_model}' sul dispositivo '{device}'...")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=target_model,
        device=device
    )

    print("Modello caricato con successo.")
    return sentiment_pipeline