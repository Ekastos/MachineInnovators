# app.py

import gradio as gr
import torch
import csv
import os
from datetime import datetime
from src.model import load_sentiment_pipeline
from src import config

# --- 1. SETUP E CARICAMENTO MODELLO ---

print("Avvio dell'applicazione...")


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda:0"
    return "cpu"


device = get_device()
print(f"Utilizzo del dispositivo: {device}")

# --- LOGICA DI SELEZIONE DEL MODELLO ---
# Definiamo il percorso dove ci aspettiamo il modello trainato
FINE_TUNED_DIR = "./fine_tuned_model"

model_to_load = None

# Controlliamo se la cartella esiste e contiene file
if os.path.exists(FINE_TUNED_DIR) and os.listdir(FINE_TUNED_DIR):
    print(f"✅ Trovato modello fine-tuned in: {FINE_TUNED_DIR}")
    print("Utilizzo del modello personalizzato.")
    model_to_load = FINE_TUNED_DIR
else:
    print("ℹ️ Nessun modello fine-tuned trovato (o cartella vuota).")
    print(f"Utilizzo del modello base: {config.MODEL_NAME}")
    model_to_load = None  # La funzione userà il default da config

# Carichiamo la pipeline
sentiment_pipeline = load_sentiment_pipeline(device=device, model_name=model_to_load)

# File dove salveremo le correzioni
LOG_FILE = "flagged_data_corrected.csv"

# Creiamo l'header del CSV se non esiste
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "text", "model_prediction", "user_correction"])


# --- 2. FUNZIONI LOGICHE ---

def predict(text):
    """
    Fa la previsione. Restituisce:
    1. Il dizionario per il grafico (Label)
    2. Una stringa con la label predetta (da salvare nello stato nascoso)
    3. Rende visibile il pannello di correzione
    """
    if not text:
        return None, None, gr.Column(visible=False)

    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()
    score = result['score']

    # Formattiamo per il componente Label (che vuole {Label: Score})
    output_dict = {result['label'].capitalize(): score}

    # Restituiamo: Output visivo, Stato nascosto, Rendi visibile la correzione
    return output_dict, label, gr.Column(visible=True)


def save_correction(text, model_prediction, user_correction):
    """
    Salva la correzione nel CSV.
    """
    if not user_correction:
        return "⚠️ Per favore seleziona un'etichetta corretta prima di salvare."

    timestamp = datetime.now().isoformat()

    with open(LOG_FILE, "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, text, model_prediction, user_correction])

    return f"✅ Correzione salvata! (Modello: {model_prediction} -> Utente: {user_correction})"


# --- 3. INTERFACCIA GRAFICA CON BLOCKS ---

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 📈 Analisi del Sentiment & Miglioramento Continuo")

    # Mostriamo all'utente quale modello sta usando
    if model_to_load == FINE_TUNED_DIR:
        gr.Markdown("🚀 **Status:** Utilizzo del modello *Fine-Tuned* (Personalizzato)")
    else:
        gr.Markdown("🔵 **Status:** Utilizzo del modello *Base* (Pre-addestrato)")

    gr.Markdown(
        "Inserisci un testo per analizzare il sentiment. Se il modello sbaglia, aiutaci a migliorare correggendolo qui sotto.")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Testo da analizzare", lines=4,
                                    placeholder="Es: I love this product but the shipping was slow.")
            analyze_btn = gr.Button("Analizza Sentiment", variant="primary")

        with gr.Column():
            output_label = gr.Label(label="Previsione Modello", num_top_classes=3)
            prediction_state = gr.State()

    # --- SEZIONE DI CORREZIONE (Inizialmente Nascosta) ---
    with gr.Column(visible=False) as correction_section:
        gr.Markdown("---")
        gr.Markdown("### 🔧 Il modello ha sbagliato?")
        gr.Markdown("Seleziona l'etichetta corretta qui sotto e clicca 'Salva Correzione'.")

        with gr.Row():
            correction_radio = gr.Radio(
                choices=config.LABELS,
                label="Scegli l'etichetta corretta:",
                interactive=True
            )
            save_btn = gr.Button("💾 Salva Correzione")

        status_message = gr.Markdown("")

    # --- 4. COLLEGAMENTI DEGLI EVENTI ---

    analyze_btn.click(
        fn=predict,
        inputs=input_text,
        outputs=[output_label, prediction_state, correction_section]
    )

    save_btn.click(
        fn=save_correction,
        inputs=[input_text, prediction_state, correction_radio],
        outputs=status_message
    )

if __name__ == "__main__":
    demo.launch()