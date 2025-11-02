# app.py

import gradio as gr
import torch
from src.model import load_sentiment_pipeline

# --- Logica per il caricamento del modello ---

print("Avvio dell'applicazione Gradio...")


def get_device():
    """Controlla e restituisce il dispositivo disponibile ottimale."""
    if torch.backends.mps.is_available():
        return "mps"
    # Aggiungiamo il controllo per CUDA per completezza, sebbene non usato su Mac
    elif torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


# Carichiamo il modello una sola volta all'avvio dell'app
# Questo è molto più efficiente che caricarlo ad ogni predizione.
device = get_device()
print(f"Utilizzo del dispositivo: {device}")
sentiment_pipeline = load_sentiment_pipeline(device=device)

print("Modello caricato. L'applicazione è pronta.")


# --- Funzione per l'interfaccia ---

def predict_sentiment(text):
    """
    Esegue la predizione del sentiment sul testo in input
    e formatta l'output per Gradio.
    """
    if not text:
        return {}  # Restituisce un dizionario vuoto se non c'è testo

    # La pipeline restituisce una lista con un dizionario, es: [{'label': 'positive', 'score': 0.99}]
    result = sentiment_pipeline(text)[0]

    # Formattiamo l'output come un dizionario di {etichetta: punteggio}
    # Questo è il formato che l'componente "Label" di Gradio si aspetta.
    label = result['label'].capitalize()
    score = result['score']

    return {label: score}


# --- Costruzione dell'interfaccia Gradio ---

# Usiamo il tema di default di Gradio
theme = gr.themes.Default()

# Descrizione per l'interfaccia
title = "Analisi del Sentiment per MachineInnovators Inc."
description = """
Questo strumento analizza il sentiment di un testo (in inglese) relativo ai social media,
classificandolo come Negative, Neutral, o Positive.
Il modello utilizzato è `cardiffnlp/twitter-roberta-base-sentiment-latest`.
Inserisci un testo e clicca su "Submit" per vedere il risultato.
"""

# Creiamo l'interfaccia
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Scrivi qui il tuo testo..."),
    outputs=gr.Label(num_top_classes=3, label="Risultato del Sentiment"),
    title=title,
    description=description,
    examples=[
        ["MachineInnovators Inc. is revolutionizing the AI industry!"],
        ["The new update is okay, but it could be better."],
        ["I am very disappointed with their customer service."]
    ],
    theme=theme
)

# Avviamo l'applicazione
if __name__ == "__main__":
    iface.launch()