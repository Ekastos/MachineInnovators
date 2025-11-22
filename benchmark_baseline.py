# main.py

import torch
from src.data_loader import load_sentiment_dataset
from src.model import load_sentiment_pipeline
from src.evaluate import evaluate_model


def get_device():
    """
    Controlla e restituisce il dispositivo disponibile ottimale (MPS su Mac, altrimenti CPU).
    """
    if torch.backends.mps.is_available():
        # Controlla se il backend MPS (per GPU Apple Silicon) è disponibile
        device = "mps"
    else:
        # Altrimenti, usa la CPU
        device = "cpu"

    print(f"Utilizzo del dispositivo: {device}")
    return device


def main():
    """
    Funzione principale per eseguire l'intero processo:
    1. Determina il dispositivo
    2. Carica i dati
    3. Carica il modello sul dispositivo corretto
    4. Valuta il modello
    """
    # 1. Ottieni il dispositivo ottimale
    device = get_device()

    # 2. Carica il dataset
    dataset = load_sentiment_dataset()

    # 3. Carica la pipeline del modello, specificando il dispositivo
    sentiment_pipeline = load_sentiment_pipeline(device=device)

    # 4. Valuta le performance del modello
    # La valutazione non ha bisogno di modifiche, la pipeline sa già dove eseguire i calcoli.
    evaluate_model(sentiment_pipeline, dataset)


if __name__ == "__main__":
    main()