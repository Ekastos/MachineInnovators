# src/monitor.py

import pandas as pd
import os
from sklearn.metrics import accuracy_score
from src import config
from src.data_loader import load_sentiment_dataset
from src.model import load_sentiment_pipeline
import warnings

# Ignoriamo gli avvisi di UserWarning da scikit-learn per un output più pulito
warnings.filterwarnings("ignore", category=UserWarning)

# Definiamo una soglia di accuratezza sotto la quale attivare un alert
ACCURACY_THRESHOLD = 0.65


def check_performance_drift(pipeline, new_data):
    """
    Controlla il degrado delle performance su un nuovo set di dati.
    """
    print("\n--- Controllo Performance Drift ---")

    texts = list(new_data['text'])
    true_labels_str = new_data['label'].tolist()
    true_labels_ids = [config.LABEL2ID.get(label.lower()) for label in true_labels_str]

    valid_indices = [i for i, label_id in enumerate(true_labels_ids) if label_id is not None]
    texts = [texts[i] for i in valid_indices]
    true_labels_ids = [true_labels_ids[i] for i in valid_indices]

    if not texts:
        print("Nessun dato valido per la valutazione.")
        return

    model_outputs = pipeline(texts)
    predicted_labels_str = [output['label'].lower() for output in model_outputs]
    predicted_labels_ids = [config.LABEL2ID[label] for label in predicted_labels_str]

    accuracy = accuracy_score(true_labels_ids, predicted_labels_ids)
    print(f"Accuratezza sui nuovi dati: {accuracy:.4f}")
    print(f"Soglia di accuratezza minima: {ACCURACY_THRESHOLD}")

    if accuracy < ACCURACY_THRESHOLD:
        print("ALERT: Performance Drift Rilevato! L'accuratezza è scesa sotto la soglia.")
        print("Si consiglia di avviare il retraining del modello.")
    else:
        print("OK: Le performance del modello sono stabili.")

    print("--- Fine Controllo ---")


def run_monitoring():
    """
    Funzione principale per eseguire il monitoraggio.
    """
    print("Avvio dello script di monitoraggio...")

    # --- FIX PER IL PATH DEL MODELLO ---
    # Definiamo il percorso relativo
    local_model_path = "./fine_tuned_model"

    # Controlliamo se esiste E se contiene il file config.json (segno che è un modello valido)
    model_to_use = None

    if os.path.isdir(local_model_path) and "config.json" in os.listdir(local_model_path):
        # Convertiamo in percorso ASSOLUTO per evitare errori di Hugging Face
        model_to_use = os.path.abspath(local_model_path)
        print(f"Trovato modello locale. Utilizzo percorso assoluto: {model_to_use}")
    else:
        print("Nessun modello locale trovato (o incompleto). Utilizzo il modello base da Hugging Face.")

    # Carica il modello (locale o base)
    pipeline = load_sentiment_pipeline(device="cpu", model_name=model_to_use)
    # -----------------------------------

    flagged_data = {
        'text': [
            "This new feature is kinda meh",
            "I'm not NOT happy with this.",
            "Wow, MachineInnovators is on another level! #sarcasm",
            "Just love waiting 2 hours for customer support."
        ],
        'label': [
            'negative',
            'positive',
            'negative',
            'negative'
        ]
    }
    new_data_df = pd.DataFrame(flagged_data)

    print(f"\nSimulazione del monitoraggio su {len(new_data_df)} nuovi campioni di dati.")
    check_performance_drift(pipeline, new_data_df)


if __name__ == "__main__":
    run_monitoring()