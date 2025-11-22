# tests/test_model.py

import pytest
from src.model import load_sentiment_pipeline
from src.config import LABELS


# --- FIXTURE ---
# Utilizziamo una fixture con scope="module" per caricare il modello
# una sola volta per tutti i test di questo file.
# Questo rende i test molto più veloci.
@pytest.fixture(scope="module")
def pipeline():
    # Forziamo la CPU per i test per garantire che girino ovunque (anche su GitHub Actions)
    return load_sentiment_pipeline(device="cpu")


# --- TEST ESISTENTI (MIGLIORATI) ---

def test_pipeline_structure(pipeline):
    """
    Verifica che la pipeline restituisca i dati nel formato corretto:
    [{'label': '...', 'score': ...}]
    """
    text = "Test sentence"
    result = pipeline(text)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert 'label' in result[0]
    assert 'score' in result[0]
    assert isinstance(result[0]['score'], float)


def test_labels_consistency(pipeline):
    """
    Verifica che le etichette restituite dal modello siano
    esattamente quelle definite nella nostra configurazione.
    """
    text = "Test"
    result = pipeline(text)
    prediction = result[0]['label'].lower()

    # Controlla che la label predetta sia una di quelle permesse (negative, neutral, positive)
    assert prediction in LABELS


# --- NUOVI TEST: ANTI-DEGRADO E ROBUSTEZZA ---

def test_model_performance_sanity_check(pipeline):
    """
    GOLDEN SET / ANTI-DEGRADATION TEST:
    Questo test verifica che il modello mantenga una "conoscenza di base".
    Se il modello sbaglia queste frasi ovvie, significa che è degradato
    (es. dopo un retraining sbagliato) e non deve andare in produzione.
    """
    # Dizionario di frasi inequivocabili -> etichetta attesa
    golden_examples = {
        "I absolutely love this product, it is amazing!": "positive",
        "Worst experience ever, I hate it.": "negative",
        "The package arrived on Tuesday.": "neutral",  # Fatto oggettivo
        "Excellent service and great quality.": "positive",
        "Disgusting food and rude staff.": "negative"
    }

    for text, expected_label in golden_examples.items():
        result = pipeline(text)
        predicted_label = result[0]['label'].lower()

        # Se fallisce qui, il modello è "rotto" o ha dimenticato concetti base
        assert predicted_label == expected_label, \
            f"Errore critico: '{text}' classificato come {predicted_label} invece di {expected_label}"


def test_robustness_edge_cases(pipeline):
    """
    Testa come il modello gestisce input strani o limite.
    Il modello non deve crashare (sollevare eccezioni).
    """
    edge_cases = [
        "",  # Stringa vuota
        "   ",  # Solo spazi
        "12345",  # Solo numeri
        "😂🔥👍",  # Solo emoji
        "a" * 512  # Testo molto lungo (potrebbe essere troncato, ma non deve crashare)
    ]

    for text in edge_cases:
        try:
            result = pipeline(text)
            # Deve comunque restituire una lista valida
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Il modello è crashato con l'input '{text}': {str(e)}")


def test_confidence_score(pipeline):
    """
    Verifica che lo score di confidenza sia un valore valido (tra 0 e 1).
    """
    text = "Valid text"
    result = pipeline(text)
    score = result[0]['score']

    assert 0.0 <= score <= 1.0, f"Lo score {score} è fuori dal range [0, 1]"