import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

st.set_page_config(page_title="CSV → Grafico", layout="wide", page_icon="📊")

def load_css(filepath):
    try:
        with open(filepath, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")
st.title("📊 CSV to Chart")
st.markdown("Carica un dataset, scegli colonne e tipo di grafico, esporta e ottieni il codice Python.")

def load_dataset():
    if "df_pre" in st.session_state:
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

    source = st.radio(
        "Sorgente dati",
        ["📂 Upload CSV", "📦 Libreria ISLP", "🔗 URL diretto"],
        horizontal=True
    )
    df = None

    if source == "📂 Upload CSV":
        uploaded = st.file_uploader("Carica CSV", type="csv")
        if uploaded:
            # Prova prima con virgola, poi con punto e virgola
            try:
                df = pd.read_csv(uploaded)
                # Verifica che abbia più di una colonna (altrimenti sep sbagliato)
                if df.shape[1] == 1:
                    uploaded.seek(0)  # Riporta il file all'inizio
                    df = pd.read_csv(uploaded, sep=";", decimal=",")
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, sep=";", decimal=",")

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
                df = df.replace("?", pd.NA)
                df = df.apply(pd.to_numeric, errors="ignore")
                st.session_state.filename = f"{choice}.csv"
            except ImportError:
                st.error("Libreria ISLP non installata. Esegui: pip install ISLP")
            except Exception as e:
                st.error(f"Errore nel caricamento: {e}")

    elif source == "🔗 URL diretto":
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
        url = st.text_input(
            "Incolla URL del file CSV o link Kaggle",
            placeholder="https://www.kaggle.com/datasets/user/dataset  oppure  https://raw.githubusercontent.com/.../file.csv"
        )
        st.caption("💡 Kaggle: incolla l'URL della pagina del dataset (es. kaggle.com/datasets/user/nome) | GitHub: usa il link Raw del file .csv")

        if url and st.button("Carica da URL"):
            try:
                if "kaggle.com/datasets" in url:
                    import tempfile, os

                    # ← Imposta le credenziali PRIMA di importare kaggle
                    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
                    os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]

                    import kaggle  # ← Solo dopo aver settato le env variables

                    parts = url.rstrip("/").split("/datasets/")[-1]
                    dataset_id = "/".join(parts.split("/")[:2])

                    with tempfile.TemporaryDirectory() as tmpdir:
                        kaggle.api.authenticate()
                        kaggle.api.dataset_download_files(dataset_id, path=tmpdir, unzip=True)
                        csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
                        if not csv_files:
                            st.error("Nessun file CSV trovato nel dataset Kaggle.")
                            st.stop()
                        chosen = st.selectbox("Scegli CSV:", csv_files) if len(csv_files) > 1 else csv_files[0]
                        df = pd.read_csv(os.path.join(tmpdir, chosen))
                        for col in df.select_dtypes(include="object").columns:
                            df[col] = df[col].astype(str)
                        st.session_state.filename = chosen

                else:
                    raw_text = requests.get(url, timeout=10).text
                    df = pd.read_csv(io.StringIO(raw_text))
                    if df.shape[1] == 1:
                        df = pd.read_csv(io.StringIO(raw_text), sep=";", decimal=",")
                    st.session_state.filename = url.split("/")[-1] or "dataset.csv"

                st.session_state.df = df
                st.rerun()

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

selected_file = st.session_state.get("filename", "dataset.csv")
st.success(f"✅ {selected_file} — {df.shape[0]} righe × {df.shape[1]} colonne")

with st.expander("🔍 Anteprima dati", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"{df.shape[0]} righe × {df.shape[1]} colonne")
    
    st.markdown("**Tipi di dato per colonna:**")
    dtype_df = pd.DataFrame({
        "Colonna": df.columns,
        "Tipo": df.dtypes.values.astype(str),
        "Valori unici": df.nunique().values,
        "Nulli": df.isnull().sum().values,
        "Esempio": [df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A" for col in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


st.divider()

# ── Chart config ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
chart_type = col1.selectbox("Tipo di grafico",
    ["Line", "Bar", "Scatter", "Heatmap", "Histogram", "Box", "Area", "Scatter Matrix"])

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()
cat_cols = [c for c in all_cols if c not in numeric_cols]

chart_height = st.slider("Altezza grafico (px)", min_value=300, max_value=1200, value=500, step=50)

if chart_type == "Heatmap":
    selected_cols = col2.multiselect("Colonne numeriche (heatmap)", numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))])
    x_col, y_cols, color_col = None, None, None
    use_facet = False

elif chart_type == "Histogram":
    x_col = col2.selectbox("Colonna (X)", numeric_cols)
    color_col = col3.selectbox("Colore (opzionale)", ["Nessuno"] + all_cols)
    y_cols, selected_cols = None, None
    use_facet = False

elif chart_type == "Scatter Matrix":
    selected_cols = col2.multiselect("Colonne (scatter matrix)", numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))])
    color_col = col3.selectbox("Colore per categoria (opzionale)", ["Nessuno"] + all_cols)
    x_col, y_cols = None, None
    use_facet = False

else:
    col_x, col_y = st.columns(2)
    x_col = col_x.selectbox("Asse X", all_cols, index=0)
    y_options = [c for c in numeric_cols if c != x_col]
    y_cols = [col_y.selectbox("Asse Y", y_options if y_options else numeric_cols)]
    color_col = st.selectbox("Colore per categoria (opzionale)", ["Nessuno"] + all_cols)
    selected_cols = None

    # ── Facet come opzione aggiuntiva ─────────────────────────────────────────
    use_facet = st.checkbox("🔲 Abilita Facet (suddividi il grafico per categoria)")
    facet_col, facet_row = None, None
    if use_facet:
        if not cat_cols:
            st.warning("Nessuna colonna categorica disponibile per il facet.")
            use_facet = False
        else:
            fc1, fc2 = st.columns(2)
            facet_col_sel = fc1.selectbox("Facet per colonna", ["Nessuno"] + cat_cols)
            facet_row_sel = fc2.selectbox("Facet per riga (opzionale)", ["Nessuno"] + cat_cols)
            facet_col = facet_col_sel if facet_col_sel != "Nessuno" else None
            facet_row = facet_row_sel if facet_row_sel != "Nessuno" else None

st.divider()

code_lines = [
    "import pandas as pd",
    "import plotly.express as px",
    "import plotly.graph_objects as go",
    "",
    f"df = pd.read_csv(\"{selected_file}\")",
    "",
]

fig = None
try:
    if chart_type == "Heatmap":
        if not selected_cols:
            st.warning("Seleziona almeno due colonne per la heatmap.")
            st.stop()
        corr = df[selected_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Heatmap di correlazione", height=chart_height)
        code_lines += [f"selected_cols = {selected_cols}", "corr = df[selected_cols].corr()",
                       "fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')"]

    elif chart_type == "Histogram":
        kw = {} if color_col == "Nessuno" else {"color": color_col}
        fig = px.histogram(df, x=x_col, **kw, title=f"Histogram – {x_col}", height=chart_height)
        color_str = f", color='{color_col}'" if color_col != "Nessuno" else ""
        code_lines += [f"fig = px.histogram(df, x='{x_col}'{color_str})"]

    elif chart_type == "Scatter Matrix":
        if not selected_cols or len(selected_cols) < 2:
            st.warning("Seleziona almeno due colonne.")
            st.stop()
        kw = {} if color_col == "Nessuno" else {"color": color_col}
        fig = px.scatter_matrix(df, dimensions=selected_cols, **kw,
                                title="Scatter Matrix", height=chart_height)
        fig.update_traces(diagonal_visible=False)
        color_str = f", color='{color_col}'" if color_col != "Nessuno" else ""
        code_lines += [f"fig = px.scatter_matrix(df, dimensions={selected_cols}{color_str})",
                       "fig.update_traces(diagonal_visible=False)"]

    else:
        if not y_cols:
            st.warning("Seleziona almeno una colonna per l'asse Y.")
            st.stop()

        color_kw = {} if color_col == "Nessuno" else {"color": color_col}
        color_str = f", color='{color_col}'" if color_col != "Nessuno" else ""
        df_plot = df.sort_values(by=x_col) if x_col else df
        y = y_cols[0]

        # Facet kwargs
        facet_kw = {}
        facet_str = ""
        if use_facet:
            if facet_col: facet_kw["facet_col"] = facet_col; facet_str += f", facet_col='{facet_col}'"
            if facet_row: facet_kw["facet_row"] = facet_row; facet_str += f", facet_row='{facet_row}'"

        if chart_type == "Line":
            fig = px.line(df_plot, x=x_col, y=y, **color_kw, **facet_kw,
                          title=f"Line – {y} vs {x_col}", height=chart_height)
            code_lines += [f"fig = px.line(df, x='{x_col}', y='{y}'{color_str}{facet_str})"]

        elif chart_type == "Bar":
            use_count = st.checkbox("Usa conteggio (count) come Y", value=False)
            if use_count:
                df_count = df[x_col].value_counts().reset_index()
                df_count.columns = [x_col, "count"]
                fig = px.bar(df_count, x=x_col, y="count", title=f"Bar – count di {x_col}",
                             height=chart_height)
                code_lines += [f"df_count = df['{x_col}'].value_counts().reset_index()",
                               f"df_count.columns = ['{x_col}', 'count']",
                               f"fig = px.bar(df_count, x='{x_col}', y='count')"]
            else:
                agg_func = st.selectbox("Aggregazione", ["mean", "sum", "median", "min", "max"])
                df_grouped = df.groupby(x_col)[y].agg(agg_func).reset_index()
                fig = px.bar(df_grouped, x=x_col, y=y, **color_kw, **facet_kw,
                             title=f"Bar – {agg_func}({y}) per {x_col}", height=chart_height)
                code_lines += [f"df_grouped = df.groupby('{x_col}')['{y}'].agg('{agg_func}').reset_index()",
                               f"fig = px.bar(df_grouped, x='{x_col}', y='{y}'{color_str}{facet_str})"]

        elif chart_type == "Scatter":
            fig = px.scatter(df_plot, x=x_col, y=y, **color_kw, **facet_kw,
                             title=f"Scatter – {y} vs {x_col}", height=chart_height)
            code_lines += [f"fig = px.scatter(df, x='{x_col}', y='{y}'{color_str}{facet_str})"]

        elif chart_type == "Area":
            fig = px.area(df_plot, x=x_col, y=y, **color_kw, **facet_kw,
                          title=f"Area – {y} vs {x_col}", height=chart_height)
            code_lines += [f"fig = px.area(df, x='{x_col}', y='{y}'{color_str}{facet_str})"]

        elif chart_type == "Box":
            fig = px.box(df_plot, x=x_col, y=y, **color_kw, **facet_kw,
                         title=f"Box – {y}", height=chart_height)
            code_lines += [f"fig = px.box(df, x='{x_col}', y='{y}'{color_str}{facet_str})"]

    fig.update_layout(template="plotly_white", legend=dict(orientation="h"))
    code_lines.append("fig.show()")

except Exception as e:
    st.error(f"Errore nella generazione del grafico: {e}")
    st.stop()

st.plotly_chart(fig, use_container_width=True)

st.subheader("⬇️ Esporta")
ecol1, ecol2 = st.columns(2)
try:
    img_bytes = fig.to_image(format="png", scale=2)
    ecol1.download_button("📥 Scarica PNG", data=img_bytes, file_name="grafico.png", mime="image/png")
except Exception:
    ecol1.info("Install `kaleido` per export PNG.")
html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
ecol2.download_button("📥 Scarica HTML", data=html_bytes, file_name="grafico.html", mime="text/html")

st.divider()
st.subheader("🐍 Codice Python equivalente")
st.code("\n".join(code_lines), language="python")