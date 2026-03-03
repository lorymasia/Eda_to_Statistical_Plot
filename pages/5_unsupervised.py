import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import requests
import io

def load_css(filepath):
    with open(filepath, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


st.set_page_config(page_title="Unsupervised Learning", layout="wide", page_icon="🔍")
st.title("🔍 Unsupervised Learning")
st.markdown("Clustering e riduzione dimensionale su dataset senza colonna target.")

def load_dataset():
    if "df" in st.session_state:
        st.info(f"📂 Dataset attivo: **{st.session_state.get('filename', 'dataset')}**")
        if st.button("🔄 Cambia dataset"):
            del st.session_state["df"]
            del st.session_state["filename"]
            st.rerun()
        return st.session_state.df

    source = st.radio(
        "Sorgente dati",
        ["📂 Upload CSV", "📦 Libreria ISLP", "🔗 URL diretto"],
        horizontal=True
    )
    df = None

    if source == "📂 Upload CSV":
        uploaded = st.file_uploader("Carica CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.filename = uploaded.name

    elif source == "📦 Libreria ISLP":
        islp_datasets = [
            "Advertising", "Auto", "Boston", "Caravan", "Carseats",
            "College", "Default", "Hitters", "OJ", "Portfolio",
            "Smarket", "USArrests", "Wage", "Weekly"
        ]
        choice = st.selectbox("Scegli dataset ISLP", islp_datasets)
        if st.button("Carica dataset ISLP"):
            try:
                from ISLP import load_data
                df = load_data(choice)
                st.session_state.filename = f"{choice}.csv"
            except ImportError:
                st.error("Libreria ISLP non installata. Esegui: pip install ISLP")
            except Exception as e:
                st.error(f"Errore nel caricamento: {e}")

    elif source == "🔗 URL diretto":
        url = st.text_input(
            "Incolla URL del file CSV",
            placeholder="https://raw.githubusercontent.com/.../file.csv"
        )
        st.caption("💡 Su Kaggle: apri il dataset → tre puntini → Copy URL del file .csv raw")
        if url and st.button("Carica da URL"):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                st.session_state.filename = url.split("/")[-1] or "dataset.csv"
            except requests.exceptions.HTTPError as e:
                st.error(f"Errore HTTP: {e}")
            except Exception as e:
                st.error(f"Errore: {e}")

    if df is not None:
        st.session_state.df = df
        st.rerun()

    return df

df = load_dataset()
if df is None:
    st.info("⬆️ Carica un dataset per iniziare.")
    st.stop()

st.success(f"✅ {st.session_state.get('filename', 'dataset')} — {df.shape[0]} righe × {df.shape[1]} colonne")

with st.expander("🔍 Anteprima dati", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

st.divider()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

col1, col2 = st.columns(2)
feature_cols = col1.multiselect("📥 Feature (colonne numeriche)", numeric_cols, default=numeric_cols)
task = col2.selectbox("Algoritmo", ["K-Means", "DBSCAN"])

if not feature_cols:
    st.warning("Seleziona almeno due feature.")
    st.stop()

normalize = st.checkbox("Normalizza feature (StandardScaler)", value=True)

if task == "K-Means":
    col3, col4 = st.columns(2)
    n_clusters = col3.slider("Numero di cluster (K)", min_value=2, max_value=15, value=3)
    show_elbow = col4.checkbox("Mostra Elbow Method", value=True)
else:
    col3, col4 = st.columns(2)
    eps = col3.slider("eps (raggio)", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
    min_samples = col4.slider("min_samples", min_value=2, max_value=20, value=5)

col5, col6 = st.columns(2)
viz_method = col5.selectbox("Visualizzazione 2D", ["PCA", "t-SNE"])
color_col = col6.selectbox("Colonna colore aggiuntiva (opzionale)", ["Nessuno"] + all_cols)

st.divider()

if st.button("🚀 Esegui clustering", use_container_width=True):
    try:
        data = df[feature_cols].dropna()
        X_scaled = StandardScaler().fit_transform(data) if normalize else data.values

        if task == "K-Means" and show_elbow:
            inertias = []
            k_range = range(2, 12)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
            fig_elbow = px.line(x=list(k_range), y=inertias, markers=True,
                                labels={"x": "K", "y": "Inertia"}, title="Elbow Method")
            fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red",
                                annotation_text=f"K={n_clusters}")
            st.plotly_chart(fig_elbow, use_container_width=True)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) if task == "K-Means" else DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

        data = data.copy()
        data["Cluster"] = labels.astype(str)
        labels_display = np.where(labels == -1, "Rumore", (labels + 1).astype(str))
        data = data.copy()
        data["Cluster"] = labels_display
        unique_labels = set(labels)
        if len(unique_labels) > 1 and not (unique_labels == {-1}):
            try:
                # Escludi i punti rumore (label == -1) per DBSCAN
                mask = labels != -1
                sil = silhouette_score(X_scaled[mask], labels[mask])
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    sil = silhouette_score(X_scaled[mask], labels[mask])
                    st.metric("Silhouette Score", f"{sil:.4f}")
                    if task == "DBSCAN":
                        st.caption(f"ℹ️ Calcolato su {mask.sum()} punti, esclusi {(~mask).sum()} noise points")
                else:
                    st.warning("Troppo pochi punti validi per calcolare il Silhouette Score.")
            except Exception:
                pass


        if task == "DBSCAN":
            n_noise = (labels == -1).sum()
            st.info(f"Cluster trovati: {len(set(labels)) - (1 if -1 in labels else 0)} | Noise points: {n_noise}")

        st.subheader(f"📊 Visualizzazione {viz_method}")
        if viz_method == "PCA":
            coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
        else:
            perplexity = min(30, len(X_scaled) - 1)
            coords = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X_scaled)

        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        plot_df["Cluster"] = labels_display

        if color_col != "Nessuno" and color_col in df.columns:
            plot_df[color_col] = df[color_col].values[:len(plot_df)]
            fig_s = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", symbol=color_col,
                               title=f"{viz_method} – Cluster", hover_data=[color_col])
        else:
            fig_s = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", title=f"{viz_method} – Cluster")
        st.plotly_chart(fig_s, use_container_width=True)

        st.subheader("📋 Statistiche per cluster")
        st.dataframe(data.groupby("Cluster")[feature_cols].mean().round(3), use_container_width=True)

        code = f"""import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import {'KMeans' if task == 'K-Means' else 'DBSCAN'}
from sklearn.decomposition import PCA
import plotly.express as px

df = pd.read_csv("your_file.csv")
X = df[{feature_cols}].dropna()
X_scaled = StandardScaler().fit_transform(X)
model = {'KMeans(n_clusters=' + str(n_clusters) + ', random_state=42, n_init=10)' if task == 'K-Means' else 'DBSCAN(eps=' + str(eps) + ', min_samples=' + str(min_samples) + ')'}
labels = model.fit_predict(X_scaled)
coords = PCA(n_components=2).fit_transform(X_scaled)
plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
plot_df["Cluster"] = labels.astype(str)
px.scatter(plot_df, x="PC1", y="PC2", color="Cluster").show()"""

        st.divider()
        st.subheader("🐍 Codice Python equivalente")
        st.code(code, language="python")

        csv_out = data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Scarica CSV con etichette cluster", data=csv_out,
                           file_name="clustered_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Errore: {e}")