import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import requests
import io

st.set_page_config(page_title="Data Preprocessing", layout="wide", page_icon="🛠️")

def load_css(filepath):
    try:
        with open(filepath, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")
st.title("🛠️ Data Preprocessing & Feature Engineering")
st.markdown("Esplora, pulisci e trasforma il tuo dataset. Le modifiche restano attive su tutto il sito.")

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
            df = pd.read_csv(uploaded)
            df = df.replace("?", pd.NA)
            df = df.apply(pd.to_numeric, errors="ignore")
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
                df = df.replace("?", pd.NA)
                df = df.apply(pd.to_numeric, errors="ignore")
                st.session_state.filename = f"{choice}.csv"
            except ImportError:
                st.error("Libreria ISLP non installata. Esegui: pip install ISLP")
            except Exception as e:
                st.error(f"Errore nel caricamento: {e}")

    elif source == "🔗 URL diretto":
        url = st.text_input("Incolla URL del file CSV",
                            placeholder="https://raw.githubusercontent.com/.../file.csv")
        st.caption("💡 Su Kaggle: apri il dataset → tre puntini → Copy URL del file .csv raw")
        if url and st.button("Carica da URL"):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                df = df.replace("?", pd.NA)
                df = df.apply(pd.to_numeric, errors="ignore")
                st.session_state.filename = url.split("/")[-1] or "dataset.csv"
            except requests.exceptions.HTTPError as e:
                st.error(f"Errore HTTP: {e}")
            except Exception as e:
                st.error(f"Errore: {e}")

    if df is not None:
        st.session_state.df = df
        st.rerun()

    return df

df_loaded = load_dataset()
if df_loaded is None:
    st.info("⬆️ Carica un dataset per iniziare.")
    st.stop()

if "df_pre" not in st.session_state or st.session_state.get("pre_filename") != st.session_state.get("filename"):
    st.session_state.df_pre = df_loaded.copy()
    st.session_state.pre_filename = st.session_state.get("filename")
    st.session_state.code_log = []

df = st.session_state.df_pre
code_log = st.session_state.code_log

st.success(f"✅ {st.session_state.get('filename', 'dataset')} — {df.shape[0]} righe × {df.shape[1]} colonne")
st.caption("💡 Le modifiche applicate qui saranno visibili anche nelle pagine Supervised e Unsupervised.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Esplorazione", "🧹 Pulizia", "🔧 Trasformazioni", "⬇️ Export"])

with tab1:
    st.subheader("Anteprima dataset")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Tipi di dato per colonna")
    dtype_df = pd.DataFrame({
        "Colonna": df.columns,
        "Tipo": df.dtypes.values.astype(str),
        "Valori unici": df.nunique().values,
        "Esempio": [df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A" for col in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.subheader("Statistiche descrittive")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Valori nulli per colonna")
    null_df = pd.DataFrame({"Colonna": df.columns, "Nulli": df.isnull().sum().values,
                             "% Nulli": (df.isnull().mean().values * 100).round(2)})
    if null_df["Nulli"].sum() == 0:
        st.success("✅ Nessun valore nullo nel dataset.")
    else:
        st.dataframe(null_df[null_df["Nulli"] > 0], use_container_width=True)
        fig_null = px.bar(null_df[null_df["Nulli"] > 0], x="Colonna", y="% Nulli",
                          title="Percentuale valori nulli", text_auto=True)
        st.plotly_chart(fig_null, use_container_width=True)

    st.subheader("Distribuzione colonne")
    col_explore = st.selectbox("Seleziona colonna", df.columns, key="explore_col")
    if pd.api.types.is_numeric_dtype(df[col_explore]):
        fig_dist = px.histogram(df, x=col_explore, marginal="box", title=f"Distribuzione — {col_explore}")
    else:
        vc = df[col_explore].value_counts().reset_index()
        vc.columns = [col_explore, "count"]
        fig_dist = px.bar(vc, x=col_explore, y="count", title=f"Distribuzione — {col_explore}")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Heatmap correlazione")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Correlazione tra feature numeriche")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Servono almeno 2 colonne numeriche per la heatmap.")

with tab2:
    st.subheader("🔁 Rimozione duplicati")
    n_dup = df.duplicated().sum()
    st.write(f"Righe duplicate trovate: **{n_dup}**")
    if n_dup > 0 and st.button("Rimuovi duplicati"):
        st.session_state.df_pre = st.session_state.df_pre.drop_duplicates().reset_index(drop=True)
        code_log.append("df = df.drop_duplicates().reset_index(drop=True)")
        st.success(f"✅ Rimossi {n_dup} duplicati.")
        st.rerun()
    st.divider()
    st.subheader("🕳️ Gestione valori nulli")
    null_cols = df.columns[df.isnull().any()].tolist()
    if not null_cols:
        st.success("✅ Nessun valore nullo.")
    else:
        col_null = st.selectbox("Colonna con nulli", null_cols, key="null_col")
        strategy = st.selectbox("Strategia", ["Rimuovi righe", "Fill con media", "Fill con mediana",
                                               "Fill con moda", "Fill con valore custom"])
        custom_val = None
        if strategy == "Fill con valore custom":
            custom_val = st.text_input("Valore custom")
        if st.button("Applica gestione nulli"):
            d = st.session_state.df_pre
            if strategy == "Rimuovi righe":
                st.session_state.df_pre = d.dropna(subset=[col_null]).reset_index(drop=True)
                code_log.append(f"df = df.dropna(subset=['{col_null}']).reset_index(drop=True)")
            elif strategy == "Fill con media":
                st.session_state.df_pre[col_null] = d[col_null].fillna(d[col_null].mean())
                code_log.append(f"df['{col_null}'] = df['{col_null}'].fillna(df['{col_null}'].mean())")
            elif strategy == "Fill con mediana":
                st.session_state.df_pre[col_null] = d[col_null].fillna(d[col_null].median())
                code_log.append(f"df['{col_null}'] = df['{col_null}'].fillna(df['{col_null}'].median())")
            elif strategy == "Fill con moda":
                st.session_state.df_pre[col_null] = d[col_null].fillna(d[col_null].mode()[0])
                code_log.append(f"df['{col_null}'] = df['{col_null}'].fillna(df['{col_null}'].mode()[0])")
            elif strategy == "Fill con valore custom" and custom_val is not None:
                try:
                    val = float(custom_val) if pd.api.types.is_numeric_dtype(d[col_null]) else custom_val
                except ValueError:
                    val = custom_val
                st.session_state.df_pre[col_null] = d[col_null].fillna(val)
                code_log.append(f"df['{col_null}'] = df['{col_null}'].fillna({repr(val)})")
            st.success(f"✅ Gestione nulli applicata su '{col_null}'.")
            st.rerun()
    st.divider()
    st.subheader("🗑️ Rimozione colonne")
    cols_to_drop = st.multiselect("Colonne da rimuovere", df.columns.tolist(), key="drop_cols")
    if cols_to_drop and st.button("Rimuovi colonne selezionate"):
        st.session_state.df_pre = st.session_state.df_pre.drop(columns=cols_to_drop)
        code_log.append(f"df = df.drop(columns={cols_to_drop})")
        st.success(f"✅ Rimosse: {', '.join(cols_to_drop)}")
        st.rerun()
    st.divider()
    st.subheader("✏️ Rinomina colonna")
    col_rename = st.selectbox("Colonna da rinominare", df.columns.tolist(), key="rename_col")
    new_name = st.text_input("Nuovo nome")
    if new_name and st.button("Rinomina"):
        st.session_state.df_pre = st.session_state.df_pre.rename(columns={col_rename: new_name})
        code_log.append(f"df = df.rename(columns={{'{col_rename}': '{new_name}'}})")
        st.success(f"✅ Rinominata '{col_rename}' → '{new_name}'")
        st.rerun()

with tab3:
    st.subheader("🔤 Encoding categoriche → numeriche")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        st.info("Nessuna colonna categorica trovata.")
    else:
        col_enc = st.selectbox("Colonna da encodare", cat_cols, key="enc_col")
        enc_type = st.selectbox("Tipo di encoding", ["Label Encoding", "One-Hot Encoding"], key="enc_type")
        if st.button("Applica encoding"):
            d = st.session_state.df_pre
            if enc_type == "Label Encoding":
                le = LabelEncoder()
                st.session_state.df_pre[col_enc] = le.fit_transform(d[col_enc].astype(str))
                code_log.append(f"from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\ndf['{col_enc}'] = le.fit_transform(df['{col_enc}'].astype(str))")
            else:
                dummies = pd.get_dummies(d[col_enc], prefix=col_enc)
                st.session_state.df_pre = pd.concat([d.drop(columns=[col_enc]), dummies], axis=1)
                code_log.append(f"dummies = pd.get_dummies(df['{col_enc}'], prefix='{col_enc}')\ndf = pd.concat([df.drop(columns=['{col_enc}']), dummies], axis=1)")
            st.success(f"✅ Encoding applicato su '{col_enc}'.")
            st.rerun()
    st.divider()
    st.subheader("📏 Normalizzazione / Scaling")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("Nessuna colonna numerica trovata.")
    else:
        cols_scale = st.multiselect("Colonne da scalare", num_cols, key="scale_cols")
        scale_type = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="scale_type")
        if cols_scale and st.button("Applica scaling"):
            scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
            scaler = scalers[scale_type]
            st.session_state.df_pre[cols_scale] = scaler.fit_transform(st.session_state.df_pre[cols_scale])
            code_log.append(f"from sklearn.preprocessing import {scale_type}\nscaler = {scale_type}()\ndf[{cols_scale}] = scaler.fit_transform(df[{cols_scale}])")
            st.success(f"✅ {scale_type} applicato.")
            st.rerun()
    st.divider()
    st.subheader("📉 Log Transform")
    num_cols2 = df.select_dtypes(include="number").columns.tolist()
    if num_cols2:
        col_log = st.selectbox("Colonna", num_cols2, key="log_col")
        if st.button("Applica Log Transform"):
            d = st.session_state.df_pre
            if (d[col_log] <= 0).any():
                st.warning("⚠️ Valori ≤ 0: verrà usato log1p.")
                st.session_state.df_pre[col_log] = np.log1p(d[col_log])
                code_log.append(f"df['{col_log}'] = np.log1p(df['{col_log}'])")
            else:
                st.session_state.df_pre[col_log] = np.log(d[col_log])
                code_log.append(f"df['{col_log}'] = np.log(df['{col_log}'])")
            st.success(f"✅ Log transform applicato su '{col_log}'.")
            st.rerun()
    st.divider()
    st.subheader("🪣 Binning")
    num_cols3 = df.select_dtypes(include="number").columns.tolist()
    if num_cols3:
        col_bin = st.selectbox("Colonna da binnare", num_cols3, key="bin_col")
        n_bins = st.slider("Numero di bin", min_value=2, max_value=10, value=3, key="n_bins")
        bin_labels_input = st.text_input("Etichette bin (opzionale, separate da virgola)", placeholder="es. basso,medio,alto")
        if st.button("Applica Binning"):
            d = st.session_state.df_pre
            labels = None
            if bin_labels_input.strip():
                labels = [l.strip() for l in bin_labels_input.split(",")]
                if len(labels) != n_bins:
                    st.error(f"Inserisci esattamente {n_bins} etichette.")
                    st.stop()
            new_col = f"{col_bin}_bin"
            st.session_state.df_pre[new_col] = pd.cut(d[col_bin], bins=n_bins, labels=labels)
            code_log.append(f"df['{new_col}'] = pd.cut(df['{col_bin}'], bins={n_bins}, labels={labels})")
            st.success(f"✅ Binning applicato: nuova colonna '{new_col}'.")
            st.rerun()
    st.divider()
    st.subheader("🔄 Cambio tipo colonna")
    col_cast = st.selectbox("Colonna", df.columns.tolist(), key="cast_col")
    st.caption(f"Tipo attuale: `{df[col_cast].dtype}`")
    cast_type = st.selectbox("Converti in", ["int", "float", "str", "bool"], key="cast_type")
    if st.button("Applica conversione tipo"):
        try:
            type_map = {"int": int, "float": float, "str": str, "bool": bool}
            st.session_state.df_pre[col_cast] = st.session_state.df_pre[col_cast].astype(type_map[cast_type])
            code_log.append(f"df['{col_cast}'] = df['{col_cast}'].astype({cast_type})")
            st.success(f"✅ '{col_cast}' convertita in {cast_type}.")
            st.rerun()
        except Exception as e:
            st.error(f"Errore: {e}")

with tab4:
    st.subheader("📋 Dataset attuale")
    st.dataframe(st.session_state.df_pre.head(100), use_container_width=True)
    st.caption(f"{st.session_state.df_pre.shape[0]} righe × {st.session_state.df_pre.shape[1]} colonne")
    st.divider()
    csv_out = st.session_state.df_pre.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Scarica dataset preprocessato (.csv)", data=csv_out,
                       file_name="preprocessed_dataset.csv", mime="text/csv", use_container_width=True)
    st.divider()
    st.subheader("🐍 Codice Python di tutte le trasformazioni")
    if not code_log:
        st.info("Nessuna trasformazione applicata ancora.")
    else:
        full_code = "import pandas as pd\nimport numpy as np\n\n"
        full_code += f"df = pd.read_csv(\"{st.session_state.get('filename', 'dataset.csv')}\")\n\n"
        full_code += "\n".join(code_log)
        st.code(full_code, language="python")
    st.divider()
    if st.button("🔄 Reset preprocessing (ricarica originale)", use_container_width=True):
        st.session_state.df_pre = st.session_state.df.copy()
        st.session_state.code_log = []
        st.rerun()