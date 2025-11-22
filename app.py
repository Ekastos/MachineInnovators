import gradio as gr
import csv
import os
from datetime import datetime

# Il file dove salveremo i dati etichettati a mano
LOG_FILE = "human_labeled_data.csv"

# Le etichette possibili
LABELS = ["negative", "neutral", "positive"]

# Inizializza il file CSV con l'header se non esiste
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "text", "human_label"])


def save_label(text, label):
    """
    Questa funzione non usa un modello AI.
    Scrive semplicemente l'input e l'etichetta fornita dall'utente nel CSV.
    """
    if not text or not label:
        return "Errore: Testo ed etichetta non possono essere vuoti."

    timestamp = datetime.now().isoformat()

    with open(LOG_FILE, "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, text, label])

    # Restituisce un messaggio di conferma e pulisce gli input per il prossimo campione
    return f"Campione salvato con successo!", "", None


with gr.Blocks() as demo:
    gr.Markdown("# Strumento di Annotazione Dati per Sentiment Analysis")
    gr.Markdown("Inserisci un testo, seleziona l'etichetta di sentiment corretta e clicca 'Salva Etichetta'.")

    text_input = gr.Textbox(lines=5, label="Testo da etichettare")
    label_input = gr.Radio(choices=LABELS, label="Seleziona il Sentiment Corretto")

    save_button = gr.Button("Salva Etichetta")

    confirmation_output = gr.Label(label="Stato")

    # Quando il bottone viene cliccato, chiama la funzione save_label
    save_button.click(
        fn=save_label,
        inputs=[text_input, label_input],
        # L'output va al messaggio di conferma e pulisce i campi di input
        outputs=[confirmation_output, text_input, label_input]
    )

if __name__ == "__main__":
    demo.launch()