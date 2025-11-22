# 📄 Documentazione del Progetto MLOps: Monitoraggio della Reputazione Online

**Azienda:** MachineInnovators Inc.
**Progetto:** Sentiment Analysis Pipeline & Continuous Monitoring
**Data:** 22 Novembre 2025

---

## 1. Introduzione e Obiettivo del Progetto
MachineInnovators Inc. necessita di un sistema scalabile per il monitoraggio della reputazione sui social media. L'obiettivo di questo progetto è stato sviluppare un'architettura **MLOps (Machine Learning Operations)** completa che non si limiti alla semplice previsione del sentiment, ma che integri automazione, deploy continuo e meccanismi di miglioramento costante del modello tramite feedback umano.

Il sistema permette di classificare automaticamente i feedback degli utenti in **Positivi**, **Neutri** o **Negativi**, con un ciclo di vita del software gestito interamente tramite pipeline CI/CD.

---

## 2. Scelte Progettuali e Architetturali

### 2.1 Selezione del Modello: Transizione a Transformer (RoBERTa)
Sebbene le specifiche iniziali suggerissero l'uso di *FastText*, è stata effettuata una scelta architetturale orientata alle performance moderne per il NLP (Natural Language Processing).
*   **Scelta:** Modello `cardiffnlp/twitter-roberta-base-sentiment-latest`.
*   **Motivazione:** RoBERTa (Robustly optimized BERT approach) è un modello basato su Transformer pre-addestrato su milioni di tweet. Rispetto a FastText, offre una comprensione semantica superiore, gestendo efficacemente il contesto, lo slang, le emoji e il sarcasmo tipici dei social media.
*   **Risultato:** Accuratezza Baseline del **~74%** sul dataset di test *TweetEval*.

### 2.2 Strategia "Human-in-the-Loop" (HITL)
Per garantire l'adattabilità del modello nel tempo, è stata implementata una strategia di **Active Learning**.
*   **Interfaccia:** Un'applicazione web (Gradio) permette agli utenti non solo di visualizzare il risultato, ma di correggerlo se errato.
*   **Flusso Dati:** Le correzioni manuali vengono salvate in un dataset dedicato (`flagged_data_corrected.csv`) che funge da *Ground Truth* dinamica per le fasi di ri-addestramento.

### 2.3 Infrastruttura CI/CD
L'automazione è gestita tramite **GitHub Actions**, scelta per la sua integrazione nativa con il repository.
*   **CI (Continuous Integration):** Esecuzione automatica di test unitari e di integrazione ad ogni push.
*   **CD (Continuous Deployment):** Sincronizzazione automatica del codice con l'ambiente di produzione su **Hugging Face Spaces**.
*   **Scheduled Monitoring:** Utilizzo di CRON jobs per l'esecuzione periodica di script di monitoraggio.

> **Nota Metodologica:** Attualmente, il workflow automatizzato su GitHub Actions esegue i test sul **modello base** presente nel repository.
> Poiché il modello *fine-tuned* (generato localmente) non viene caricato su GitHub per limiti di storage (Git LFS), la pipeline CI non vi ha accesso diretto.
> In uno scenario di produzione Enterprise, questo limite verrebbe superato integrando un **Model Registry** esterno (es. AWS S3, MLflow o Hugging Face Hub) da cui la pipeline scaricherebbe automaticamente l'ultima versione del modello retrainato prima di eseguire il monitoraggio.

---

## 3. Dettagli Implementativi

Il progetto è strutturato in moduli logici distinti:

### Fase 1: Core e Benchmark (`src/` e `benchmark_baseline.py`)
*   Sono stati sviluppati script modulari per il caricamento dati (`data_loader.py`), la gestione del modello (`model.py`) e la valutazione (`evaluate.py`).
*   Lo script `benchmark_baseline.py` valuta il modello corrente sul dataset standard, fornendo una metrica di riferimento per rilevare eventuali degradazioni.

### Fase 2: Testing Automatizzato (`tests/`)
Per garantire la stabilità, sono stati implementati test con **pytest**:
*   **Test Strutturali:** Verificano i formati di input/output della pipeline.
*   **Sanity Check ("Golden Set"):** Un test comportamentale che verifica il modello su un piccolo set di frasi inequivocabili (es. "I hate this" deve essere negativo). Questo impedisce il deploy di modelli corrotti o degradati.

### Fase 3: Pipeline di Retraining Sicuro (`retrain.py`)
Lo script di retraining gestisce il *Fine-Tuning* del modello sui dati di feedback.
*   **Prevenzione del Catastrophic Forgetting:** Durante gli esperimenti, è stato osservato che un retraining aggressivo su pochi dati portava a un crollo dell'accuratezza (dal 74% al 47%).
*   **Soluzione:** È stato configurato un **Learning Rate conservativo** (es. `1e-5` o inferiore) e un numero ridotto di epoche. Questo permette al modello di apprendere le nuove sfumature senza dimenticare la conoscenza linguistica generale.

### Fase 4: Monitoraggio (`src/monitor.py`)
Uno script dedicato simula la valutazione del modello su nuovi flussi di dati. Se l'accuratezza scende sotto una soglia critica (es. 65%), il sistema notifica un *Performance Drift*, segnalando la necessità di intervento o retraining.

---

## 4. Risultati Ottenuti

### Metriche Quantitative
*   **Accuratezza Baseline:** 0.74 (74%)
*   **Precisione Media:** ~0.74
*   **Recall Media:** ~0.75

### Risultati Qualitativi
1.  **Robustezza:** La pipeline CI impedisce efficacemente l'introduzione di codice che rompe le funzionalità core.
2.  **Adattabilità:** L'applicazione è in grado di caricare automaticamente un modello *fine-tuned* se presente localmente, permettendo un aggiornamento fluido delle capacità predittive.
3.  **Stabilità del Retraining:** Grazie al tuning degli iperparametri, il processo di retraining si è dimostrato stabile, mantenendo le performance generali inalterate pur integrando i nuovi casi d'uso.

---

## 5. Conclusioni e Sviluppi Futuri

Il progetto consegna a MachineInnovators Inc. un'infrastruttura MLOps funzionante e scalabile. L'integrazione tra RoBERTa, GitHub Actions e Hugging Face fornisce un equilibrio ottimale tra prestazioni allo stato dell'arte e facilità di gestione operativa.

**Possibili estensioni future (Enterprise Roadmap):**
*   Integrazione di un **Model Registry** esterno (es. MLflow o AWS S3) per versionare i file binari dei modelli retrainati.
*   Implementazione di un sistema di alerting (Email/Slack) collegato al workflow di monitoraggio.