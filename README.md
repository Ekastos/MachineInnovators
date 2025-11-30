---
title: Analisi del Sentiment - MachineInnovators
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# 🤖 Monitoraggio della Reputazione Online - MachineInnovators Inc.

Benvenuti nel repository del progetto MLOps per il monitoraggio della reputazione aziendale. Questo sistema implementa una pipeline **end-to-end** per l'analisi del sentiment, integrando pratiche avanzate di DevOps e Machine Learning.

🔗 **Link Utili:**
*   **Live Demo (Hugging Face Spaces):** [Clicca qui per vedere l'App](https://huggingface.co/spaces/LorenzoIVF/Sentiment-Analyzer-MachineInnovators)
*   **Dataset Originale:** TweetEval (Sentiment)

---

## 🏗 Architettura e Funzionalità

Il progetto non si limita a un semplice modello predittivo, ma costruisce un intero ecosistema **MLOps**:

### 1. Modello di Analisi del Sentiment (`src/model.py`)
Abbiamo scelto **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) rispetto a FastText per la sua superiore capacità di comprendere il contesto, lo slang e le sfumature tipiche dei social media.
*   **Baseline Accuracy:** ~74% sul dataset TweetEval.

### 2. Human-in-the-Loop & Feedback (`app.py`)
L'interfaccia utente, sviluppata con **Gradio Blocks**, include un meccanismo di feedback attivo:
*   L'utente può analizzare un testo.
*   Se la predizione è errata, appare una sezione di correzione.
*   Il feedback umano viene salvato in un dataset (`flagged_data_corrected.csv`) per migliorare il modello.

### 3. Pipeline di Retraining Automatizzato (`retrain.py`)
Uno script dedicato permette al modello di evolversi nel tempo:
*   Carica i feedback corretti dagli utenti.
*   Esegue il **Fine-Tuning** del modello RoBERTa utilizzando un *Learning Rate* conservativo per evitare il *Catastrophic Forgetting*.
*   L'applicazione rileva automaticamente se è presente un modello *fine-tuned* locale e lo utilizza al posto di quello base.

### 4. CI/CD e Monitoraggio Continuo (GitHub Actions)
Il repository include workflow automatizzati in `.github/workflows/`:
*   **CI Pipeline:** Ad ogni push, vengono eseguiti test unitari e comportamentali ("Golden Set") per garantire che il modello non degradi su frasi fondamentali.
*   **Sync to Hub:** Deploy automatico dell'applicazione su Hugging Face Spaces.
*   **Scheduled Monitoring:** Un job CRON giornaliero esegue `src/monitor.py` per rilevare eventuali degradazioni delle performance (*Data Drift*).

> **Nota sul Monitoraggio:** Il workflow automatizzato esegue i test sul modello base presente nel repository. Per monitorare il modello *fine-tuned* in produzione, in uno scenario enterprise verrebbe utilizzato un Model Registry esterno (es. S3/MLflow) per l'archiviazione dei pesi del modello.

---

## 🚀 Come Eseguire il Progetto in Locale

Per replicare l'ambiente e testare le funzionalità di retraining:

1.  **Clonare il repository:**
    ```bash
    git clone https://github.com/LorenzoIVF/machine-innovators-reputation.git
    cd machine-innovators-reputation
    ```

2.  **Installare le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Eseguire l'App (Interfaccia Web):**
    ```bash
    python app.py
    ```

4.  **Eseguire il Benchmark (Valutazione Baseline):**
    ```bash
    python benchmark_baseline.py
    ```

5.  **Avviare il Retraining (se sono presenti dati di feedback):**
    ```bash
    python retrain.py
    ```

---

## 📂 Struttura del Repository

*   `src/`: Codice sorgente per il caricamento dati, modello e valutazione.
*   `tests/`: Test unitari e comportamentali (Pytest).
*   `fine_tuned_model/`: Cartella di output per il modello ri-addestrato (generata localmente).
*   `app.py`: Applicazione Gradio (Frontend & Logica).
*   `benchmark_baseline.py`: Script per valutare le performance generali.
*   `retrain.py`: Pipeline di fine-tuning.
*   `.github/workflows/`: Configurazioni CI/CD.