import streamlit as st

def load_css(filepath):
    try:
        with open(filepath, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

st.set_page_config(page_title="Guida", page_icon="📖", layout="wide")

# ─── Session state ────────────────────────────────────────────────────────────
if "open_item" not in st.session_state:
    st.session_state.open_item = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🤖 Algoritmi"

def toggle(key):
    st.session_state.open_item = None if st.session_state.open_item == key else key

def set_tab(tab_name):
    st.session_state.active_tab = tab_name
    st.session_state.open_item = None  # chiude tutto quando cambi tab

def accordion(key, title, content):
    is_open = st.session_state.open_item == key
    icon = "🔽" if is_open else "▶️"
    if st.button(f"{icon}  {title}", key=f"btn_{key}", use_container_width=True):
        toggle(key)
        st.rerun()
    if is_open:
        with st.container(border=True):
            st.markdown(content)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📖 Guida al Sito")
st.markdown("Una spiegazione semplice di tutto quello che trovi in questo sito: algoritmi, metriche e concetti.")

# ─── Tab navigation custom ───────────────────────────────────────────────────
TABS = ["🤖 Algoritmi", "📊 Metriche", "⚙️ Concetti ML", "📂 Dataset", "🔑 Kaggle API"]

tab_cols = st.columns(len(TABS))
for i, tab_name in enumerate(TABS):
    with tab_cols[i]:
        is_active = st.session_state.active_tab == tab_name
        btn_type = "primary" if is_active else "secondary"
        if st.button(tab_name, key=f"tab_{i}", use_container_width=True, type=btn_type):
            set_tab(tab_name)
            st.rerun()

st.divider()
active = st.session_state.active_tab

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ALGORITMI
# ════════════════════════════════════════════════════════════════════════════════
if active == "🤖 Algoritmi":
    st.header("Algoritmi di Machine Learning")

    st.subheader("🔵 Classificazione")
    st.markdown("Gli algoritmi di classificazione cercano di **assegnare un'etichetta** (categoria) a ogni campione.")

    accordion("decision_tree", "🌳 Decision Tree — Albero decisionale", """
**Cos'è?**  
Immagina una serie di domande tipo *"il valore è maggiore di X?"* — il modello costruisce un albero
di decisioni binarie fino ad arrivare alla risposta finale.

**Perché usarlo?**  
- Facile da interpretare visivamente  
- Non richiede normalizzazione dei dati  
- Ottimo per capire *quali feature* contano di più  

**Quando funziona peggio?**  
Tende all'overfitting (impara troppo bene i dati di training) se l'albero è troppo profondo.
    """)

    accordion("rf_class", "🌲 Random Forest", """
**Cos'è?**  
Un insieme (*forest*) di tanti alberi decisionali addestrati su sottoinsiemi casuali dei dati.
La predizione finale è la risposta più votata tra tutti gli alberi.

**Perché usarlo?**  
- Più robusto e preciso del singolo Decision Tree  
- Riduce l'overfitting grazie alla media tra più alberi  
- Gestisce bene dati con molte feature  

**Quando funziona peggio?**  
Meno interpretabile del singolo albero, e più lento su dataset enormi.
    """)

    accordion("logistic", "📈 Logistic Regression — Regressione Logistica", """
**Cos'è?**  
Nonostante il nome, è un classificatore. Calcola la **probabilità** che un campione appartenga
a una classe usando una curva sigmoidale (a forma di S).

**Perché usarlo?**  
- Semplice, veloce e interpretabile  
- Ottimo punto di partenza (*baseline*) per qualsiasi problema di classificazione  
- Funziona molto bene quando le classi sono linearmente separabili  

**Quando funziona peggio?**  
Non cattura relazioni non lineari tra le feature.
    """)

    accordion("svm_class", "🔵 SVM — Support Vector Machine", """
**Cos'è?**  
Trova il **confine ottimale** (iperpiano) che separa le classi con il margine più grande possibile.
Con il *kernel trick* può separare anche classi non linearmente separabili.

**Perché usarlo?**  
- Molto efficace con dati ad alta dimensionalità  
- Robusto agli outlier  

**Quando funziona peggio?**  
Lento su dataset molto grandi e difficile da interpretare.
    """)

    accordion("knn", "🤝 KNN — K-Nearest Neighbors", """
**Cos'è?**  
Per classificare un nuovo campione, guarda i **K punti più vicini** nel dataset e assegna
la classe più frequente tra loro.

**Perché usarlo?**  
- Zero fase di training (impara tutto al momento della predizione)  
- Intuitivo e semplice  

**Quando funziona peggio?**  
Lento nella predizione su dati grandi, e sensibile alla scala delle feature (usa la normalizzazione!).
    """)

    st.subheader("🟠 Regressione")
    st.markdown("Gli algoritmi di regressione cercano di **predire un valore numerico continuo**.")

    accordion("linear_reg", "📏 Linear Regression — Regressione Lineare", """
**Cos'è?**  
Trova la retta (o iperpiano) che meglio approssima i dati minimizzando la somma degli errori al quadrato.
La formula è: `y = m·x + q`

**Perché usarlo?**  
- Semplicissimo e interpretabile (coefficienti diretti)  
- Veloce da addestrare  
- Ottima baseline per problemi di regressione  

**Quando funziona peggio?**  
Non cattura relazioni non lineari.
    """)

    accordion("ridge", "🔷 Ridge Regression", """
**Cos'è?**  
È una regressione lineare con una **penalizzazione** (L2) sui coefficienti troppo grandi.
Questo riduce l'overfitting e stabilizza il modello.

**Perché usarlo?**  
- Quando hai molte feature correlate tra loro  
- Quando la regressione lineare semplice va in overfitting  

**Parametro chiave:** `alpha` — più è alto, più i coefficienti vengono "schiacciati" verso zero.
    """)

    accordion("rf_reg", "🌲 Random Forest Regressor", """
**Cos'è?**  
Stesso principio del Random Forest per classificazione, ma invece di votare una classe,
fa la **media delle predizioni** numeriche di ogni albero.

**Perché usarlo?**  
- Cattura relazioni non lineari  
- Robusto agli outlier  
- Non richiede normalizzazione  
    """)

    accordion("svr", "🔵 SVR — Support Vector Regression", """
**Cos'è?**  
Versione per regressione della SVM. Cerca la funzione che approssima i dati
mantenendo gli errori entro una soglia `ε` (epsilon).

**Perché usarlo?**  
- Efficace con dati ad alta dimensionalità  
- Buon compromesso tra flessibilità e generalizzazione  
    """)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — METRICHE
# ════════════════════════════════════════════════════════════════════════════════
elif active == "📊 Metriche":
    st.header("Metriche di Valutazione")

    st.subheader("🔵 Metriche di Classificazione")
    col1, col2 = st.columns(2)

    with col1:
        accordion("accuracy", "✅ Accuracy", """
**Cos'è?**  
La percentuale di predizioni corrette sul totale dei campioni.

**Formula:** `Accuracy = predizioni corrette / totale campioni`

**Quando usarla?**  
Quando le classi sono **bilanciate** (simile numero di campioni per classe).

⚠️ Con classi sbilanciate può essere fuorviante: un modello che predice sempre la classe maggioritaria
può avere alta accuracy ma essere inutile.
        """)

        accordion("precision", "🎯 Precision", """
**Cos'è?**  
Di tutti i campioni che il modello ha detto *"positivo"*, quanti lo erano davvero?

**Formula:** `Precision = Veri Positivi / (Veri Positivi + Falsi Positivi)`

**Quando è importante?**  
Quando i **falsi positivi** sono costosi — es. sistema antispam: meglio non bloccare email
legittime che lasciar passare qualche spam.
        """)

        accordion("recall", "🔍 Recall", """
**Cos'è?**  
Di tutti i campioni realmente positivi, quanti ne ha trovati il modello?

**Formula:** `Recall = Veri Positivi / (Veri Positivi + Falsi Negativi)`

**Quando è importante?**  
Quando i **falsi negativi** sono costosi — es. diagnosi medica: meglio non perdere nessun malato,
anche a costo di qualche falso allarme.
        """)

    with col2:
        accordion("f1", "⚖️ F1 Score", """
**Cos'è?**  
La media armonica di Precision e Recall. È un buon indicatore unico quando
vuoi bilanciare entrambe le metriche.

**Formula:** `F1 = 2 · (Precision · Recall) / (Precision + Recall)`

**Quando usarlo?**  
Con classi sbilanciate, dove accuracy da sola non basta.
Un F1 vicino a 1 = modello eccellente.
        """)

        accordion("confusion_matrix", "🗺️ Confusion Matrix", """
**Cos'è?**  
Una tabella che mostra per ogni classe reale quante volte il modello ha predetto ogni classe.
La diagonale principale rappresenta le predizioni corrette.

**Come leggerla?**  
- Diagonale piena = modello preciso  
- Valori fuori diagonale = errori (confusioni tra classi)  

Ti dice esattamente *dove* sbaglia il modello, non solo *quanto* sbaglia.
        """)

        accordion("roc_auc", "📈 ROC Curve & AUC", """
**Cos'è?**  
La curva ROC mostra il trade-off tra **True Positive Rate** (Recall) e **False Positive Rate**
al variare della soglia di classificazione.

**AUC (Area Under the Curve):**  
- `AUC = 1.0` → modello perfetto  
- `AUC = 0.5` → modello casuale (inutile)  
- `AUC > 0.8` → modello buono  

Utile per confrontare modelli indipendentemente dalla soglia scelta.
        """)

    st.subheader("🟠 Metriche di Regressione")
    col3, col4 = st.columns(2)

    with col3:
        accordion("mse", "📐 MSE — Mean Squared Error", """
**Cos'è?**  
La media degli errori al quadrato tra valori reali e predetti.

**Formula:** `MSE = media( (y_reale - y_predetto)² )`

**Caratteristica:**  
Penalizza molto gli errori grandi (li eleva al quadrato).
Sensibile agli outlier. Più basso è, meglio è.
        """)

        accordion("rmse", "📏 RMSE — Root Mean Squared Error", """
**Cos'è?**  
È semplicemente la radice quadrata dell'MSE.

**Perché usarlo?**  
Ha lo stesso significato dell'MSE ma è espresso nella **stessa unità di misura** del target,
quindi è più intuitivo da interpretare.

Es. se stai predicendo prezzi in €, RMSE ti dice *"mediamente sbaglio di X €"*.
        """)

    with col4:
        accordion("mae", "📦 MAE — Mean Absolute Error", """
**Cos'è?**  
La media degli errori in valore assoluto.

**Formula:** `MAE = media( |y_reale - y_predetto| )`

**Differenza con MSE:**  
Non eleva al quadrato, quindi è **più robusto agli outlier**.
Tutti gli errori pesano uguale, grandi o piccoli.
        """)

        accordion("r2", "🏆 R² Score — Coefficiente di Determinazione", """
**Cos'è?**  
Misura quanto il modello **spiega la varianza** del target rispetto a un modello che
predice sempre la media.

**Interpretazione:**  
- `R² = 1.0` → predizione perfetta  
- `R² = 0.0` → equivale a predire sempre la media (inutile)  
- `R² < 0` → peggio della media (modello sbagliato)  

È la metrica più immediata per capire la qualità complessiva di un modello di regressione.
        """)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONCETTI ML
# ════════════════════════════════════════════════════════════════════════════════
elif active == "⚙️ Concetti ML":
    st.header("Concetti Fondamentali di ML")

    accordion("train_test", "🔀 Train / Test Split", """
**Cos'è?**  
Prima di addestrare un modello, si divide il dataset in due parti:
- **Training set**: i dati che il modello usa per imparare
- **Test set**: i dati che il modello non ha mai visto, usati per valutarne le prestazioni reali

**Perché?**  
Se valutassimo il modello sugli stessi dati usati per addestrarlo, sembrerebbe sempre perfetto —
ma in realtà potrebbe aver semplicemente *memorizzato* i dati senza capire nulla.

**Proporzione tipica:** 70-80% training, 20-30% test.
    """)

    accordion("overfitting", "📉 Overfitting e Underfitting", """
**Overfitting:**  
Il modello impara *troppo bene* i dati di training, inclusi i rumori casuali.
Risultato: ottimo sul training, pessimo sul test.
→ Come uno studente che memorizza le risposte senza capire il concetto.

**Underfitting:**  
Il modello è troppo semplice per catturare i pattern nei dati.
Risultato: pessimo sia sul training che sul test.
→ Come uno studente che non ha studiato affatto.

**Obiettivo:** trovare il giusto equilibrio tra i due.
    """)

    accordion("normalization", "📊 Normalizzazione / Standardizzazione", """
**Cos'è?**  
Trasformare le feature in modo che abbiano scala simile.

**StandardScaler** (standardizzazione):  
Trasforma ogni feature in modo che abbia media 0 e deviazione standard 1.  
`x_scaled = (x - media) / deviazione_standard`

**Perché è importante?**  
Algoritmi come KNN e SVM sono sensibili alla scala: una feature con valori 0-10000
domina su una con valori 0-1. La normalizzazione mette tutte le feature sullo stesso piano.

Decision Tree e Random Forest **non ne hanno bisogno**.
    """)

    accordion("hyperparams", "🎛️ Iperparametri", """
**Cos'è?**  
Sono i parametri che **tu** imposti prima dell'addestramento e che controllano il comportamento
del modello — il modello non li impara dai dati.

**Esempi:**  
- `n_estimators` in Random Forest → quanti alberi usare  
- `C` e `kernel` in SVM → quanto penalizzare gli errori e che tipo di confine usare  
- `alpha` in Ridge → quanto penalizzare i coefficienti grandi  
- `n_neighbors` in KNN → quanti vicini considerare  

Sceglierli bene fa una grande differenza nelle prestazioni.
    """)

    accordion("cross_val", "🔁 Cross-Validation", """
**Cos'è?**  
Invece di fare un solo train/test split, si divide il dataset in K parti (fold).
Si addestra K volte, ogni volta usando una parte diversa come test set.

**Perché?**  
Dà una stima più affidabile delle prestazioni reali del modello,
perché mediamo i risultati su K esperimenti diversi invece di uno solo.

La **K-Fold Cross-Validation** con K=5 o K=10 è lo standard.
    """)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET
# ════════════════════════════════════════════════════════════════════════════════
elif active == "📂 Dataset":
    st.header("Come Caricare un Dataset")
    st.markdown("Il sito supporta tre modalità diverse per fornire i dati agli algoritmi.")

    accordion("upload", "💾 File Locale (Upload)", """
**Cos'è?**  
Puoi caricare direttamente un file salvato sul tuo computer.

**Formati supportati:**  
- `.csv` → il più comune, valori separati da virgola  
- `.xlsx` → file Excel  

**Come usarlo?**  
1. Clicca su *"Browse files"* o trascina il file nell'area di upload  
2. Il sito legge automaticamente le colonne  
3. Scegli quale colonna usare come **target** (la variabile da predire)  

**Quando usarlo?**  
Quando hai già un tuo dataset pronto da analizzare.
    """)

    accordion("islp", "📚 Libreria ISLP", """
**Cos'è?**  
ISLP (*Introduction to Statistical Learning with Python*) è una libreria che include
dataset didattici usati nel famoso libro omonimo.

**Come usarlo?**  
1. Seleziona la modalità ISLP  
2. Scegli il nome del dataset dal menu  
3. Il sito lo carica automaticamente senza bisogno di file esterni  

**Perché è utile?**  
I dataset ISLP sono puliti, ben documentati e perfetti per esplorare
e confrontare algoritmi senza preoccuparsi della qualità dei dati.
    """)

    accordion("url", "🔗 URL / Link Esterno", """
**Cos'è?**  
Puoi incollare direttamente un link a un file CSV disponibile online.

**Esempi di fonti:**  
- GitHub (usa il link *Raw* del file)  
- Kaggle (link diretto al CSV tramite API)  
- Qualsiasi URL che punti direttamente a un file `.csv`  

**Come usarlo?**  
1. Seleziona la modalità URL  
2. Incolla il link nel campo di testo  
3. Il sito scarica e carica il dataset automaticamente  

**Nota:** l'URL deve puntare direttamente al file, non a una pagina web.  
Es. `https://raw.githubusercontent.com/utente/repo/main/data.csv` ✅
    """)

elif active == "🔑 Kaggle API":
    st.header("🔑 Configurazione Kaggle API")
    st.markdown("Per scaricare dataset direttamente da Kaggle è necessario configurare le credenziali API una sola volta.")

    accordion("kaggle_step1", "1️⃣ Ottieni la tua API Key", """
**Dove trovarla:**  
1. Vai su [kaggle.com](https://kaggle.com) e accedi al tuo account  
2. Clicca sulla tua **foto profilo** in alto a destra → **Settings**  
3. Scorri fino alla sezione **API**  
4. Clicca **Create New Token**  

Verrà scaricato automaticamente un file chiamato `kaggle.json` con questo contenuto:
```json
{"username": "tuousername", "key": "abc123xyz..."}
```
5. Dopo aver ottenuto le credenziali sostituiscile al file secrets.toml
"""
    )