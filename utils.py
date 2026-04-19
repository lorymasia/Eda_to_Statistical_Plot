import streamlit as st
import pandas as pd
import requests
import io


ISLP_DATASETS = [
    "Advertising", "Auto", "Boston", "Caravan", "Carseats",
    "College", "Default", "Hitters", "OJ", "Portfolio",
    "Smarket", "USArrests", "Wage", "Weekly"
]

CHART_TYPES = ["Line", "Bar", "Scatter", "Heatmap", "Histogram", "Box", "Area", "Scatter Matrix"]

ML_MODELS_CLASSIFICATION = {
    "Random Forest": "RandomForestClassifier",
    "Logistic Regression": "LogisticRegression",
    "SVM": "SVC",
    "KNN": "KNeighborsClassifier"
}

ML_MODELS_REGRESSION = {
    "Random Forest": "RandomForestRegressor",
    "Linear Regression": "LinearRegression",
    "Ridge Regression": "Ridge",
    "SVR": "SVR"
}


def load_css(filepath="style.css"):
    try:
        with open(filepath, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def load_csv_file(uploaded):
    encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    separators = [(",", "."), (";", ","), ("\t", ".")]
    
    for enc in encodings_to_try:
        for sep, dec in separators:
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, sep=sep, decimal=dec, encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    
    uploaded.seek(0)
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=";", decimal=",")
    return df


def load_dataset_from_url(url):
    if "kaggle.com/datasets" in url:
        import tempfile
        import os

        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]

        import kaggle

        parts = url.rstrip("/").split("/datasets/")[-1]
        dataset_id = "/".join(parts.split("/")[:2])

        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset_id, path=tmpdir, unzip=True)
            csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
            if not csv_files:
                return None, None, "Nessun file CSV trovato nel dataset Kaggle."
            chosen = csv_files[0] if len(csv_files) == 1 else None
            df = pd.read_csv(os.path.join(tmpdir, chosen))
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].astype(str)
            return df, chosen, None
    else:
        try:
            raw_text = requests.get(url, timeout=10).text
            df = pd.read_csv(io.StringIO(raw_text))
            if df.shape[1] == 1:
                df = pd.read_csv(io.StringIO(raw_text), sep=";", decimal=",")
            filename = url.split("/")[-1] or "dataset.csv"
            return df, filename, None
        except Exception as e:
            return None, None, str(e)


def render_dataset_status(show_processed=True):
    if show_processed and "df_pre" in st.session_state:
        st.info(f"📂 Dataset attivo: **{st.session_state.get('filename', 'dataset')}** *(con modifiche preprocessing)*")
        col_a, col_b = st.columns(2)
        if col_a.button("🔄 Cambia dataset"):
            for key in ["df", "df_pre", "filename", "pre_filename", "code_log"]:
                st.session_state.pop(key, None)
            st.rerun()
        if col_b.button("↩️ Usa dataset originale"):
            st.session_state.pop("df_pre", None)
            st.rerun()
        return st.session_state.df_pre

    elif "df" in st.session_state:
        st.info(f"📂 Dataset attivo: **{st.session_state.get('filename', 'dataset')}**")
        if st.button("🔄 Cambia dataset"):
            for key in ["df", "df_pre", "filename", "pre_filename", "code_log"]:
                st.session_state.pop(key, None)
            st.rerun()
        return st.session_state.df

    return None


def render_dataset_source_selection():
    source = st.radio(
        "Sorgente dati",
        ["📂 Upload CSV", "📦 Libreria ISLP", "🔗 URL diretto"],
        horizontal=True
    )
    df = None
    filename = None

    if source == "📂 Upload CSV":
        uploaded = st.file_uploader("Carica CSV", type="csv")
        if uploaded:
            df = load_csv_file(uploaded)
            filename = uploaded.name

    elif source == "📦 Libreria ISLP":
        choice = st.selectbox("Scegli dataset ISLP", ISLP_DATASETS)
        if st.button("Carica dataset ISLP"):
            try:
                from ISLP import load_data
                df = load_data(choice)
                df = df.replace("?", pd.NA)
                df = df.apply(pd.to_numeric, errors="ignore")
                filename = f"{choice}.csv"
            except ImportError:
                st.error("Libreria ISLP non installata. Esegui: pip install ISLP")
            except Exception as e:
                st.error(f"Errore nel caricamento: {e}")

    elif source == "🔗 URL diretto":
        with st.expander("🔑 Come configurare Kaggle API", expanded=False):
            st.markdown("""
                ### Per scaricareataset da Kaggle sono necessari 3 passaggi:

                **1. Ottieni la tua API Key**
                - Vai su [kaggle.com](https://kaggle.com) → clicca sulla tua foto profilo → **Settings**
                - Scorri fino alla sezione **API** → clicca **Create New Token**
                - Verrà scaricato un file `kaggle.json` con le tue credenziali

                **2. Crea il file di configurazione**

                Nella cartella del progetto cerca il file `.streamlit/secrets.toml` e incolla:
                ```toml
                [kaggle]
                username = "il_tuo_username"
                key = "la_tua_api_key"
            """)
        url = st.text_input(
            "Incolla URL del file CSV o link Kaggle",
            placeholder="https://www.kaggle.com/datasets/user/dataset  oppure  https://raw.githubusercontent.com/.../file.csv"
        )
        st.caption("💡 Kaggle: incolla l'URL della pagina del dataset (es. kaggle.com/datasets/user/nome) | GitHub: usa il link Raw del file .csv")

        if url and st.button("Carica da URL"):
            df, filename, error = load_dataset_from_url(url)
            if error:
                st.error(f"Errore: {error}")
            elif df is not None:
                st.session_state.df = df
                st.session_state.filename = filename
                st.rerun()

    if df is not None:
        st.session_state.df = df
        st.session_state.filename = filename
        st.rerun()

    return df


def get_active_dataset(show_processed=True):
    df = render_dataset_status(show_processed)
    if df is not None:
        return df

    df = render_dataset_source_selection()
    return df


def validate_target(df, target_col):
    if target_col not in df.columns:
        return False, "Colonna target non trovata"

    n_unique = df[target_col].nunique()
    if n_unique == 0:
        return False, "Colonna target vuota"
    if n_unique == 1:
        return False, "Il target ha un solo valore unico"
    if n_unique == len(df):
        return False, "Il target ha tanti valori quante le righe (identificatore?)"

    return True, None


def render_kaggle_help():
    with st.expander("🔑 Come configurare Kaggle API", expanded=False):
        st.markdown("""
            ### Per scaricare dataset da Kaggle sono necessari 3 passaggi:

            **1. Ottieni la tua API Key**
            - Vai su [kaggle.com](https://kaggle.com) → clicca sulla tua foto profilo → **Settings**
            - Scorri fino alla sezione **API** → clicca **Create New Token**
            - Verrà scaricato un file `kaggle.json` con le tue credenziali

            **2. Crea il file di configurazione**

            Nella cartella del progetto cerca il file `.streamlit/secrets.toml` e incolla:
            ```toml
            [kaggle]
            username = "il_tuo_username"
            key = "la_tua_api_key"
        """)