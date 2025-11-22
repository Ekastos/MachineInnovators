# src/evaluate.py

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src import config


def evaluate_model(sentiment_pipeline, dataset, sample_size=1000):
    """
    Valuta le performance del modello su un campione del test set.
    """
    print(f"\nInizio della valutazione su un campione di {sample_size} elementi...")

    test_sample = dataset['test'].shuffle(seed=42).select(range(sample_size))

    # Estrai i testi e le etichette reali
    true_labels_ids = test_sample['label']

    # Convertiamo esplicitamente l'output in una lista di stringhe
    texts = list(test_sample['text'])
    # ---------------------------

    # Ottieni le predizioni dal modello
    print("Esecuzione delle predizioni...")
    model_outputs = sentiment_pipeline(texts)

    # Estrai e normalizza le etichette predette
    predicted_labels_str = [output['label'].lower() for output in model_outputs]
    predicted_labels_ids = [config.LABEL2ID[label] for label in predicted_labels_str]

    # Calcola e stampa le metriche
    accuracy = accuracy_score(true_labels_ids, predicted_labels_ids)
    print(f"\n--- Report di Valutazione ---")
    print(f"Accuratezza sul campione del test set: {accuracy:.4f}")

    print("\nReport di Classificazione Dettagliato:")
    print(classification_report(
        true_labels_ids,
        predicted_labels_ids,
        target_names=config.LABELS
    ))
