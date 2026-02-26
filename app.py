import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import os

st.set_page_config(page_title="CSV → Plot", layout="wide", page_icon="📊")

st.title("📊 CSV to Chart")
st.markdown("Carica uno o più file CSV, scegli colonne e tipo di grafico, esporta e ottieni il codice Python.")

# ── Upload ──────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Carica uno o più file CSV",
    type="csv",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⬆️ Carica almeno un file CSV per iniziare.")
    st.stop()

# ── Load & merge ─────────────────────────────────────────────────────────────
dfs = {}
for f in uploaded_files:
    try:
        df = pd.read_csv(f)
        dfs[f.name] = df
    except Exception as e:
        st.error(f"Errore nel file {f.name}: {e}")

if not dfs:
    st.stop()

st.success(f"✅ {len(dfs)} file caricato/i: {', '.join(dfs.keys())}")

# File selector
selected_file = st.selectbox("Seleziona file da visualizzare", list(dfs.keys()))
df = dfs[selected_file]

with st.expander("🔍 Anteprima dati", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"{df.shape[0]} righe × {df.shape[1]} colonne")

st.divider()

# ── Chart config ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

chart_type = col1.selectbox(
    "Tipo di grafico",
    ["Line", "Bar", "Scatter", "Heatmap", "Histogram", "Box", "Area"]
)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

if chart_type == "Heatmap":
    selected_cols = col2.multiselect(
        "Colonne numeriche (heatmap)",
        numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))]
    )
    x_col, y_col, color_col = None, None, None
elif chart_type == "Histogram":
    x_col = col2.selectbox("Colonna (X)", numeric_cols)
    color_col = col3.selectbox("Colore (opzionale)", ["Nessuno"] + all_cols)
    y_col, selected_cols = None, None
else:
    x_col = col2.selectbox("Asse X", all_cols)
    y_cols = col3.multiselect(
        "Asse Y (anche multiplo)",
        numeric_cols,
        default=[numeric_cols[0]] if numeric_cols else []
    )
    color_col = st.selectbox("Colore per categoria (opzionale)", ["Nessuno"] + all_cols)
    y_col, selected_cols = y_cols, None

st.divider()

# ── Build figure ──────────────────────────────────────────────────────────────
fig = None
code_lines = [
    "import pandas as pd",
    "import plotly.express as px",
    "import plotly.graph_objects as go",
    "",
    f'df = pd.read_csv("{selected_file}")',
    "",
]

try:
    if chart_type == "Heatmap":
        if not selected_cols:
            st.warning("Seleziona almeno due colonne per la heatmap.")
            st.stop()
        corr = df[selected_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Heatmap di correlazione"
        )
        code_lines += [
            f"selected_cols = {selected_cols}",
            "corr = df[selected_cols].corr()",
            "fig = px.imshow(corr, text_auto=\'.2f\', color_continuous_scale=\'RdBu_r\', title=\'Heatmap di correlazione\')",
        ]

    elif chart_type == "Histogram":
        kw = {} if color_col == "Nessuno" else {"color": color_col}
        fig = px.histogram(df, x=x_col, **kw, title=f"Histogram – {x_col}")
        color_str = f", color=\'{color_col}\'" if color_col != "Nessuno" else ""
        code_lines += [f"fig = px.histogram(df, x=\'{x_col}\'{color_str})"]

    else:
        if not y_cols:
            st.warning("Seleziona almeno una colonna per l'asse Y.")
            st.stop()

        color_kw = {} if color_col == "Nessuno" else {"color": color_col}
        color_str = f", color=\'{color_col}\'" if color_col != "Nessuno" else ""

        if len(y_cols) == 1:
            y = y_cols[0]
            if chart_type == "Line":
                fig = px.line(df, x=x_col, y=y, **color_kw, title=f"Line – {y} vs {x_col}")
                code_lines += [f"fig = px.line(df, x=\'{x_col}\', y=\'{y}\'{color_str})"]
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y, **color_kw, title=f"Bar – {y} vs {x_col}")
                code_lines += [f"fig = px.bar(df, x=\'{x_col}\', y=\'{y}\'{color_str})"]
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y, **color_kw, title=f"Scatter – {y} vs {x_col}")
                code_lines += [f"fig = px.scatter(df, x=\'{x_col}\', y=\'{y}\'{color_str})"]
            elif chart_type == "Area":
                fig = px.area(df, x=x_col, y=y, **color_kw, title=f"Area – {y} vs {x_col}")
                code_lines += [f"fig = px.area(df, x=\'{x_col}\', y=\'{y}\'{color_str})"]
            elif chart_type == "Box":
                fig = px.box(df, x=x_col, y=y, **color_kw, title=f"Box – {y}")
                code_lines += [f"fig = px.box(df, x=\'{x_col}\', y=\'{y}\'{color_str})"]
        else:
            # Multiple Y columns → overlay traces
            fig = go.Figure()
            for y in y_cols:
                if chart_type in ["Line", "Area"]:
                    fig.add_trace(go.Scatter(x=df[x_col], y=df[y], mode="lines", name=y,
                                             fill="tozeroy" if chart_type == "Area" else None))
                elif chart_type == "Bar":
                    fig.add_trace(go.Bar(x=df[x_col], y=df[y], name=y))
                elif chart_type == "Scatter":
                    fig.add_trace(go.Scatter(x=df[x_col], y=df[y], mode="markers", name=y))
                elif chart_type == "Box":
                    fig.add_trace(go.Box(y=df[y], name=y))
            fig.update_layout(title=f"{chart_type} – {x_col}")
            code_lines += [
                "fig = go.Figure()",
                f"for y in {y_cols}:",
                f"    fig.add_trace(go.Scatter(x=df[\'{x_col}\'], y=df[y], mode=\'lines\', name=y))",
            ]

    fig.update_layout(template="plotly_white", legend=dict(orientation="h"))
    code_lines.append("fig.show()")

except Exception as e:
    st.error(f"Errore nella generazione del grafico: {e}")
    st.stop()

# ── Show chart ────────────────────────────────────────────────────────────────
st.plotly_chart(fig, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("⬇️ Esporta")
ecol1, ecol2 = st.columns(2)

# PNG export
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

# HTML export
html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
ecol2.download_button(
    label="📥 Scarica HTML",
    data=html_bytes,
    file_name="grafico.html",
    mime="text/html"
)

# ── Python code ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("🐍 Codice Python equivalente")
st.code("\n".join(code_lines), language="python")
