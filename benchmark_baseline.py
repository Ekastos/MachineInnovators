# main.py

import torch
import os
from src.data_loader import load_sentiment_dataset
from src.model import load_sentiment_pipeline
from src.evaluate import evaluate_model
from src import config


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda:0"
    return "cpu"


def main():
    """
    Benchmark Script:
    Valuta il modello corrente (Base o Fine-Tuned) sul dataset originale TweetEval.
    Serve a verificare che il retraining non abbia causato degrado (Catastrophic Forgetting).
    """
    device = get_device()

    # 1. Logica di selezione del modello
    # Cerchiamo se esiste il modello retrainato
    local_model_path = "./fine_tuned_model"
    model_to_use = None

    # Verifica robusta: deve esistere la cartella E il file config.json
    if os.path.isdir(local_model_path) and "config.json" in os.listdir(local_model_path):
        model_to_use = os.path.abspath(local_model_path)
        print(f"\n📢 ATTENZIONE: Trovato modello Fine-Tuned locale!")
        print(f"Stiamo valutando il modello personalizzato in: {model_to_use}")
    else:
        print(f"\nℹ️ Nessun modello locale trovato.")
        print(f"Stiamo valutando il modello BASE originale: {config.MODEL_NAME}")

    # 2. Carica il dataset originale (TweetEval)
    # Questo è il tuo standard di riferimento (Ground Truth generale)
    dataset = load_sentiment_dataset()

    # 3. Carica la pipeline con il modello scelto
    sentiment_pipeline = load_sentiment_pipeline(device=device, model_name=model_to_use)

    # 4. Valuta le performance
    print("\n--- Inizio Benchmark ---")
    if model_to_use:
        print("Obiettivo: Verificare che l'accuratezza non sia peggiorata rispetto al 74% del modello base.")

    evaluate_model(sentiment_pipeline, dataset)


if __name__ == "__main__":
    main()