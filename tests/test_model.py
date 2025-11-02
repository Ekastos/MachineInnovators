# tests/test_model.py

import pytest
from src.model import load_sentiment_pipeline


def test_pipeline_loading():
    """Verifica che la pipeline si carichi senza errori."""
    pipeline = load_sentiment_pipeline()
    assert pipeline is not None


def test_pipeline_prediction():
    """Verifica che la pipeline produca una predizione nel formato atteso."""
    pipeline = load_sentiment_pipeline()
    text = "This is a great product!"
    result = pipeline(text)

    # Ci aspettiamo una lista con un dizionario
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'label' in result[0]
    assert 'score' in result[0]
    assert result[0]['label'].lower() in ['negativo', 'neutro', 'positivo']