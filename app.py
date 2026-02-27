import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

st.set_page_config(page_title="CSV → Grafico", layout="wide", page_icon="📊")
st.title("📊 CSV to Chart")
st.markdown("Carica un dataset, scegli colonne e tipo di grafico, esporta e ottieni il codice Python.")

# ── Funzione caricamento dataset ──────────────────────────────────────────────
def load_dataset():
    # Se il dataset è già in sessione, lo usa direttamente
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

    # Salva in sessione appena caricato
    if df is not None:
        st.session_state.df = df
        st.rerun()

    return df


# ── Caricamento ───────────────────────────────────────────────────────────────
df = load_dataset()
if df is None:
    st.info("⬆️ Carica un dataset per iniziare.")
    st.stop()

selected_file = st.session_state.get("filename", "dataset.csv")
st.success(f"✅ {selected_file} — {df.shape[0]} righe × {df.shape[1]} colonne")

with st.expander("🔍 Anteprima dati", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"{df.shape[0]} righe × {df.shape[1]} colonne")

st.divider()

# ── Chart config ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

chart_type = col1.selectbox(
    "Tipo di grafico",
    ["Line", "Bar", "Scatter", "Heatmap", "Histogram", "Box", "Area"]
)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

st.markdown("""
    <style>
    div[data-testid="stMultiSelect"] input {
        pointer-events: none !important;
        caret-color: transparent !important;
        user-select: none !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="input"] {
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True)

if chart_type == "Heatmap":
    selected_cols = col2.multiselect("Colonne numeriche (heatmap)", numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))])
    x_col, y_cols, color_col = None, None, None
elif chart_type == "Histogram":
    x_col = col2.selectbox("Colonna (X)", numeric_cols)
    color_col = col3.selectbox("Colore (opzionale)", ["Nessuno"] + all_cols)
    y_cols, selected_cols = None, None
else:
    col_x, col_y = st.columns(2)
    x_col = col_x.selectbox("Asse X", all_cols, index=0)
    y_cols = [col_y.selectbox("Asse Y", [c for c in numeric_cols if c != x_col], index=0)]
    color_col = st.selectbox("Colore per categoria (opzionale)", ["Nessuno"] + all_cols)
    selected_cols = None

st.divider()

# ── Build figure ──────────────────────────────────────────────────────────────
code_lines = [
    "import pandas as pd",
    "import plotly.express as px",
    "import plotly.graph_objects as go",
    "",
    f"df = pd.read_csv('{selected_file}')",
    "",
]

fig = None
try:
    if chart_type == "Heatmap":
        if not selected_cols:
            st.warning("Seleziona almeno due colonne per la heatmap.")
            st.stop()
        corr = df[selected_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Heatmap di correlazione"
        )
        code_lines += [
            f"selected_cols = {selected_cols}",
            "corr = df[selected_cols].corr()",
            "fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')",
        ]

    elif chart_type == "Histogram":
        kw = {} if color_col == "Nessuno" else {"color": color_col}
        fig = px.histogram(df, x=x_col, **kw, title=f"Histogram – {x_col}")
        color_str = f", color='{color_col}'" if color_col != "Nessuno" else ""
        code_lines += [f"fig = px.histogram(df, x='{x_col}'{color_str})"]

    else:
        if not y_cols:
            st.warning("Seleziona almeno una colonna per l'asse Y.")
            st.stop()

        color_kw = {} if color_col == "Nessuno" else {"color": color_col}
        color_str = f", color='{color_col}'" if color_col != "Nessuno" else ""
        df_plot = df.sort_values(by=x_col) if x_col else df

        if len(y_cols) == 1:
            y = y_cols[0]
            if chart_type == "Line":
                fig = px.line(df_plot, x=x_col, y=y, **color_kw, title=f"Line – {y} vs {x_col}")
                code_lines += [f"fig = px.line(df, x='{x_col}', y='{y}'{color_str})"]
            elif chart_type == "Bar":
                use_count = st.checkbox("Usa conteggio (count) come Y", value=False)
                if use_count:
                    df_count = df[x_col].value_counts().reset_index()
                    df_count.columns = [x_col, "count"]
                    fig = px.bar(df_count, x=x_col, y="count",
                                title=f"Bar – count di {x_col}")
                    code_lines += [
                        f"df_count = df['{x_col}'].value_counts().reset_index()",
                        f"df_count.columns = ['{x_col}', 'count']",
                        f"fig = px.bar(df_count, x='{x_col}', y='count')"
                    ]
                else:
                    agg_func = st.selectbox("Aggregazione", ["mean", "sum", "median", "min", "max"])
                    df_grouped = df.groupby(x_col)[y].agg(agg_func).reset_index()
                    fig = px.bar(df_grouped, x=x_col, y=y, **color_kw,
                                title=f"Bar – {agg_func}({y}) per {x_col}")
                    code_lines += [
                        f"df_grouped = df.groupby('{x_col}')['{y}'].agg('{agg_func}').reset_index()",
                        f"fig = px.bar(df_grouped, x='{x_col}', y='{y}')"
                    ]

            elif chart_type == "Scatter":
                fig = px.scatter(df_plot, x=x_col, y=y, **color_kw, title=f"Scatter – {y} vs {x_col}")
                code_lines += [f"fig = px.scatter(df, x='{x_col}', y='{y}'{color_str})"]
            elif chart_type == "Area":
                fig = px.area(df_plot, x=x_col, y=y, **color_kw, title=f"Area – {y} vs {x_col}")
                code_lines += [f"fig = px.area(df, x='{x_col}', y='{y}'{color_str})"]
            elif chart_type == "Box":
                fig = px.box(df_plot, x=x_col, y=y, **color_kw, title=f"Box – {y}")
                code_lines += [f"fig = px.box(df, x='{x_col}', y='{y}'{color_str})"]
        else:
            fig = go.Figure()
            for y in y_cols:
                if chart_type in ["Line", "Area"]:
                    fig.add_trace(go.Scatter(
                        x=df_plot[x_col], y=df_plot[y], mode="lines", name=y,
                        fill="tozeroy" if chart_type == "Area" else None
                    ))
                elif chart_type == "Bar":
                    fig.add_trace(go.Bar(x=df_plot[x_col], y=df_plot[y], name=y))
                elif chart_type == "Scatter":
                    fig.add_trace(go.Scatter(x=df_plot[x_col], y=df_plot[y], mode="markers", name=y))
                elif chart_type == "Box":
                    fig.add_trace(go.Box(y=df_plot[y], name=y))
            fig.update_layout(title=f"{chart_type} – {x_col}")
            code_lines += [
                "fig = go.Figure()",
                f"for y in {y_cols}:",
                f"    fig.add_trace(go.Scatter(x=df['{x_col}'], y=df[y], mode='lines', name=y))",
            ]

    fig.update_layout(template="plotly_white", legend=dict(orientation="h"))
    code_lines.append("fig.show()")

except Exception as e:
    st.error(f"Errore nella generazione del grafico: {e}")
    st.stop()

# ── Mostra grafico ────────────────────────────────────────────────────────────
st.plotly_chart(fig, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("⬇️ Esporta")
ecol1, ecol2 = st.columns(2)

try:
    img_bytes = fig.to_image(format="png", scale=2)
    ecol1.download_button(
        label="📥 Scarica PNG",
        data=img_bytes,
        file_name="grafico.png",
        mime="image/png"
    )
except Exception:
    ecol1.info("Install `kaleido` per export PNG: `pip install kaleido`")

html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
ecol2.download_button(
    label="📥 Scarica HTML",
    data=html_bytes,
    file_name="grafico.html",
    mime="text/html"
)

# ── Codice Python ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🐍 Codice Python equivalente")
st.code("\n".join(code_lines), language="python")