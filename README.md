# 📊 EDA to Chart & ML Models

> Una web app completa e interattiva per l'analisi esplorativa dei dati, visualizzazione e machine learning — tutto in una sola applicazione.

Una multi-pagina web app costruita con **Streamlit** che permette di caricare un dataset, esplorarlo e preprocessarlo, generare grafici interattivi con **Plotly**, e addestrare modelli di **Machine Learning** — il tutto senza scrivere codice.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ Caratteristiche Principali

### 📊 CSV to Chart
Crea visualizzazioni interattive dai tuoi dati in pochi secondi:
- **7 tipi di grafici**: Line, Bar, Scatter, Heatmap, Histogram, Box, Area
- **Grafici a barre avanzati**: Raggruppamento (mean, sum, median, min, max) o semplice conteggio
- **Personalizzazione**: Seleziona gli assi X e Y dalle colonne disponibili
- **Colori per categorie**: Raggruppa i dati per colore
- **Export multiplo**: Salva come **PNG** (immagini) o **HTML** (interattivi)
- **Codice Python automatico**: Ottieni il codice equivalente per ogni grafico

### 🛠️ Data Preprocessing
Pulizia e trasformazione completa dei dati:
- **Esplorazione dataset**: Statistiche, valori nulli, distribuzioni, heatmap correlazioni
- **Panoramica tipi di dati**: Tipo, valori unici, esempi per ogni colonna
- **Pulizia dati**:
  - Rimozione duplicati
  - Gestione valori nulli (drop, fill mean/median/mode/custom)
  - Eliminazione colonne
  - Rinominazione colonne
- **Trasformazioni avanzate**:
  - Label Encoding & One-Hot Encoding
  - Normalizzazione: StandardScaler / MinMaxScaler / RobustScaler
  - Log Transform
  - Binning
  - Type casting colonne
- **Export CSV**: Scarica il dataset preprocessato
- **Codice Python completo**: Pipeline di preprocessing automaticamente generato

### 🤖 Supervised Learning
Addestramento e valutazione di modelli supervisionati:
- **Rilevamento automatico**: Classifica il task come classification o regression
- **Modelli di Classificazione**: Random Forest, Logistic Regression, SVM, KNN
- **Modelli di Regressione**: Random Forest, Linear Regression, Ridge, SVR
- **Metriche di Classificazione**: Accuracy, F1, Precision, Recall + Confusion Matrix
- **Metriche di Regressione**: MSE, RMSE, MAE, R² + Residual Plot + Distribuzione residui con curva normale
- **Feature Importance**: Visualizzazione importanza feature (Random Forest)
- **Export modello**: Salva il modello addestrato come **.pkl**
- **Codice Python**: Generazione automatica del codice equivalente

### 🔍 Unsupervised Learning
Clustering e riduzione dimensionalità:
- **Algoritmi di Clustering**: K-Means e DBSCAN
- **Elbow Method**: Visualizzazione per selezionare K ottimale
- **Silhouette Score**: Valutazione della qualità dei cluster
- **Visualizzazione 2D**: PCA o t-SNE
- **Statistiche cluster**: Analisi dettagliata per gruppo
- **Export cluster**: Scarica dataset con etichette di cluster come CSV
- **Codice Python**: Generazione automatica del codice equivalente

---

## 📂 Fonti Dati Supportate

Carica i tuoi dati in 3 modi:
- **📂 Upload CSV**: Carica file dal tuo computer
- **📦 Libreria ISLP**: Dataset built-in (Auto, Boston, Wage, ecc.)
- **🔗 URL diretto**: Carica da GitHub raw o link CSV pubblici

✅ **Il dataset caricato persiste** su tutte le pagine — non occorre ricaricare quando cambiate pagina.

---

## 🛠️ Requisiti

- **Python 3.8+**
- streamlit
- pandas
- plotly
- kaleido
- kaggle
- scikit-learn
- joblib
- numpy
- scipy
- requests
- ISLP

---

## 📦 Installazione

### 1. Clona il repository
```bash
git clone https://github.com/lorymasia/Eda_to_Statistical_Plot.git
cd Eda_to_Statistical_Plot
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

Avvia l'applicazione Streamlit:

```bash
streamlit run app.py
```

---

## 📁 Struttura del Progetto

```
Eda_to_Statistical_Plot/
├── app.py                      # Pagina principale: CSV to Chart
├── pages/
│   ├── 1_Preprocessing.py      # Data Preprocessing
│   ├── 2_Supervised.py         # Supervised Learning
│   └── 3_Unsupervised.py       # Unsupervised Learning
├── style.css                   # Stili personalizzati
├── requirements.txt            # Dipendenze Python
├── .streamlit/                 # Configurazione Streamlit
├── .gitignore
└── README.md                   # Questo file
```

---

## 💡 Esempi di Utilizzo

### Esempio 1: Analizzare una vendita dataset
1. Vai su **CSV to Chart**
2. Carica il tuo file CSV
3. Seleziona X: "Date", Y: "Sales", Colore: "Category"
4. Scegli "Line Chart"
5. Esporta come PNG o HTML

### Esempio 2: Pulire e addestrare un modello
1. Vai su **Data Preprocessing**
2. Carica il dataset
3. Rimuovi valori nulli e normalizza
4. Esporta il dataset pulito
5. Vai su **Supervised Learning**
6. Carica il dataset preprocessato
7. Addestra il modello Random Forest
8. Scarica il modello `.pkl`

---

## 🎯 Funzionalità Avanzate

✅ **Pipelines automatiche**: Genera codice Python pronto per essere usato  
✅ **Visualizzazioni interattive**: Plotly per grafici dinamici  
✅ **Export multiplo**: Immagini, HTML, CSV, modelli  
✅ **Zero coding**: UI intuitiva per tutti i livelli di utente  
✅ **Machine Learning ready**: Valutazione modelli e metriche complete  

---

## 🤝 Contributi

I contributi sono benvenuti! Se trovi un bug o hai suggerimenti per miglioramenti:
1. Apri un **Issue**
2. Crea un **Pull Request** con le tue modifiche

---

**Buon analisi dei dati! 📊✨**
