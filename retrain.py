# retrain.py

import pandas as pd
import os
import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importiamo le configurazioni
from src import config

# --- CONFIGURAZIONE ---
CSV_FILE = "flagged_data_corrected.csv"
NEW_MODEL_DIR = "./fine_tuned_model"
BASE_MODEL = config.MODEL_NAME


def preprocess_function(examples, tokenizer):
    """Tokenizza i testi per il modello."""
    return tokenizer(examples['text'], truncation=True, padding=True)


def compute_metrics(eval_pred):
    """Calcola l'accuratezza durante la valutazione."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def load_corrected_data():
    """
    Carica i dati dal CSV delle correzioni e li prepara per il training.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Errore: Il file {CSV_FILE} non esiste. Raccogli prima qualche dato con l'app!")
        return None

    # Leggiamo il CSV
    df = pd.read_csv(CSV_FILE)

    # Filtriamo eventuali righe vuote o incomplete
    df = df.dropna(subset=['text', 'user_correction'])

    if len(df) < 5:
        print(f"Attenzione: Hai solo {len(df)} esempi. Il training potrebbe non essere efficace.")
        print("Consiglio: Raccogli almeno 10-20 correzioni prima di lanciare il retraining.")

    print(f"Trovati {len(df)} nuovi esempi per il retraining.")

    # Mappiamo le etichette stringa (es: 'positive') in ID numerici (es: 2)
    try:
        df['label'] = df['user_correction'].apply(lambda x: config.LABEL2ID[x.lower()])
    except KeyError as e:
        print(f"Errore nei dati: Trovata un'etichetta non valida nel CSV: {e}")
        print("Assicurati che il CSV contenga solo 'negative', 'neutral', 'positive'.")
        return None

    return df


def run_retraining():
    print("--- Avvio Pipeline di Retraining ---")

    # 1. Caricamento Dati
    df = load_corrected_data()
    if df is None:
        return

    # 2. Split Training/Evaluation
    if len(df) > 5:
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    else:
        train_df = df
        eval_df = df

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # 3. Preparazione Modello e Tokenizer
    print(f"Scaricamento modello base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(config.LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID
    )

    # Tokenizzazione
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_eval = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # 4. Configurazione Training
    training_args = TrainingArguments(
        output_dir="./training_output",
        learning_rate=1e-5, # Molto basso per non sconvolgere i pesi
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 5. Esecuzione Training
    print("Inizio addestramento...")
    trainer.train()

    # 6. Salvataggio
    print(f"Salvataggio del nuovo modello in: {NEW_MODEL_DIR}")
    trainer.save_model(NEW_MODEL_DIR)
    tokenizer.save_pretrained(NEW_MODEL_DIR)

    print("--- Retraining Completato! ---")
    print("Ora puoi caricare la cartella 'fine_tuned_model' su Hugging Face o usarla localmente.")


if __name__ == "__main__":
    run_retraining()