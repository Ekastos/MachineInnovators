# sentiment_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np


class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Assicurati che il modello sia in modalità valutazione
        self.model.eval()

        # Mappatura delle etichette del modello Hugging Face
        # Di solito, per questo modello, l'ordine è negativo, neutrale, positivo
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict_sentiment(self, text_input):
        """
        Prende in input una stringa o una lista di stringhe e restituisce il sentiment.
        """
        if isinstance(text_input, str):
            texts = [text_input]
        elif isinstance(text_input, list):
            texts = text_input
        else:
            raise ValueError("Input deve essere una stringa o una lista di stringhe.")

        # Tokenizzazione dell'input
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():  # Disabilita il calcolo del gradiente per l'inferenza
            output = self.model(**encoded_input)

        scores = output.logits.softmax(dim=1).numpy()  # Converti i logits in probabilità

        sentiments = []
        for i, text in enumerate(texts):
            # Trova l'indice della probabilità massima
            predicted_label_id = np.argmax(scores[i])
            # Mappa l'ID all'etichetta del sentiment
            sentiment_label = self.id2label[predicted_label_id]
            sentiment_score = scores[i][predicted_label_id]  # Punteggio di confidenza

            sentiments.append({
                "text": text,
                "sentiment": sentiment_label,
                "confidence": float(sentiment_score),
                "scores": {self.id2label[k]: float(v) for k, v in enumerate(scores[i])}
            })

        return sentiments if isinstance(text_input, list) else sentiments[0]


# Esempio di utilizzo (solo per testare localmente o nel notebook)
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Test con una singola frase
    text1 = "I love this product, it's amazing!"
    result1 = analyzer.predict_sentiment(text1)
    print(f"Sentiment for '{text1}': {result1}")

    text2 = "This is an okay product, nothing special."
    result2 = analyzer.predict_sentiment(text2)
    print(f"Sentiment for '{text2}': {result2}")

    text3 = "I absolutely hate this service, it's terrible."
    result3 = analyzer.predict_sentiment(text3)
    print(f"Sentiment for '{text3}': {result3}")

    # Test con una lista di frasi
    texts_list = [
        "The new update is fantastic!",
        "It crashes sometimes, which is annoying.",
        "I have no strong feelings about this."
    ]
    results_list = analyzer.predict_sentiment(texts_list)
    print("\nSentiment for list of texts:")
    for res in results_list:
        print(f"  - '{res['text']}': {res['sentiment']} (Confidence: {res['confidence']:.2f})")