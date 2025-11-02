# src/config.py

# Nome del modello da Hugging Face
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Nome del dataset e del subset da Hugging Face
DATASET_NAME = "tweet_eval"
DATASET_SUBSET = "sentiment"

# Mappatura delle etichette per la classificazione
LABELS = ["negative", "neutral", "positive"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}