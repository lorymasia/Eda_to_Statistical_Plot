import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc)
from scipy import stats
import joblib
import requests
import io

st.set_page_config(page_title="Supervised Learning", layout="wide", page_icon="🤖")

def load_css(filepath):
    try:
        with open(filepath, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")
st.title("🤖 Supervised Learning")
st.markdown("Carica un dataset, scegli le feature, il target e il modello da addestrare.")

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

st.success(f"✅ {st.session_state.get('filename', 'dataset')} — {df.shape[0]} righe × {df.shape[1]} colonne")

with st.expander("🔍 Anteprima dati", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

st.divider()

all_cols = df.columns.tolist()
col1, col2 = st.columns(2)
feature_cols = col1.multiselect("📥 Feature (X)", all_cols, default=all_cols[:-1])
target_col = col2.selectbox("🎯 Target (Y)", [c for c in all_cols if c not in feature_cols], index=0)

if not feature_cols:
    st.warning("Seleziona almeno una feature.")
    st.stop()

n_unique = df[target_col].nunique()
if n_unique == 0:
    st.error("La colonna target è vuota.")
    st.stop()
if n_unique == 1:
    st.error("Il target ha un solo valore unico. Seleziona una colonna con almeno 2 classi.")
    st.stop()
if n_unique == len(df):
    st.warning("⚠️ Il target ha tanti valori quante le righe. Potrebbe essere un identificatore, non un target.")

task_type = "classification" if df[target_col].dtype == object or df[target_col].nunique() <= 10 else "regression"
st.info(f"**Task rilevato:** {'📊 Classificazione' if task_type == 'classification' else '📈 Regressione'}")

col3, col4 = st.columns(2)
if task_type == "classification":
    model_name = col3.selectbox("Modello", ["Random Forest", "Logistic Regression", "SVM", "KNN"])
else:
    model_name = col3.selectbox("Modello", ["Random Forest", "Linear Regression", "Ridge Regression", "SVR"])

test_size = col4.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
normalize = st.checkbox("Normalizza feature (StandardScaler)", value=True)

with st.expander("⚙️ Opzioni avanzate"):
    use_cv = st.checkbox("Usa Cross-Validation (K-Fold)", value=False)
    if use_cv:
        cv_folds = st.slider("Numero di fold (K)", min_value=3, max_value=10, value=5)
    else:
        cv_folds = 1
    use_tuning = st.checkbox("🔍 Hyperparameter Tuning", value=False)
    if use_tuning:
        if task_type == "classification":
            if model_name == "Random Forest":
                n_estimators = st.slider("n_estimators", 10, 200, 100)
                max_depth = st.slider("max_depth (None=unlimited)", 1, 20, None)
                min_samples_split = st.slider("min_samples_split", 2, 10, 2)
            elif model_name == "KNN":
                n_neighbors = st.slider("n_neighbors", 1, 20, 5)
            elif model_name == "SVM":
                C = st.select_slider("C", [0.01, 0.1, 1, 10, 100])
                kernel = st.selectbox("kernel", ["rbf", "linear", "poly"])
            elif model_name == "Logistic Regression":
                C = st.select_slider("C", [0.01, 0.1, 1, 10, 100])
        else:
            if model_name == "Random Forest":
                n_estimators = st.slider("n_estimators", 10, 200, 100)
                max_depth = st.slider("max_depth (None=unlimited)", 1, 20, None)
                min_samples_split = st.slider("min_samples_split", 2, 10, 2)
            elif model_name == "Ridge Regression":
                alpha = st.select_slider("alpha", [0.01, 0.1, 1, 10, 100])
            elif model_name == "SVR":
                C = st.select_slider("C", [0.01, 0.1, 1, 10, 100])
                epsilon = st.slider("epsilon", 0.0, 0.5, 0.1)
    else:
        n_estimators, max_depth, min_samples_split = 100, None, 2
        n_neighbors, C, kernel, alpha, epsilon = 5, 1, "rbf", 1.0, 0.1
    chart_height = st.slider("Altezza grafici (px)", min_value=300, max_value=900, value=500, step=50)

st.divider()

if st.button("🚀 Addestra modello", use_container_width=True):
    try:
        data = df[feature_cols + [target_col]].dropna()
        X = data[feature_cols].copy()
        y = data[target_col].copy()

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        le_target = None
        if task_type == "classification" and y.dtype == object:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        models_clf = {
            "Random Forest": RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, random_state=42
            ) if use_tuning else RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=C, random_state=42
            ) if use_tuning else LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(
                C=C, kernel=kernel, probability=True, random_state=42
            ) if use_tuning else SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(
                n_neighbors=n_neighbors
            ) if use_tuning else KNeighborsClassifier()
        }
        models_reg = {
            "Random Forest": RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, random_state=42
            ) if use_tuning else RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=alpha) if use_tuning else Ridge(),
            "SVR": SVR(C=C, epsilon=epsilon) if use_tuning else SVR()
        }

        model = models_clf[model_name] if task_type == "classification" else models_reg[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cv_scores = None
        if use_cv and cv_folds > 1 and task_type == "classification":
            try:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception:
                pass
        elif use_cv and cv_folds > 1 and task_type == "regression":
            try:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception:
                pass

        st.subheader("📊 Risultati")

        if cv_scores is not None:
            st.info(f"📈 Cross-Validation ({cv_folds}-Fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if task_type == "classification":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("F1 (weighted)", f"{f1:.4f}")
            m3.metric("Precision", f"{prec:.4f}")
            m4.metric("Recall", f"{rec:.4f}")

            # Confusion Matrix
            labels = le_target.classes_.tolist()
            cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
            fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                               color_continuous_scale="Blues", title="Confusion Matrix",
                               labels=dict(x="Predicted", y="Actual"), height=chart_height)
            st.plotly_chart(fig_cm, use_container_width=True)

            # ROC-AUC
            classes = np.unique(y_test)
            n_classes = len(classes)
            try:
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)
                elif hasattr(model, "decision_function"):
                    y_score = model.decision_function(X_test)
                else:
                    y_score = None

                if y_score is not None:
                    st.subheader("📈 Curva ROC-AUC")
                    fig_roc = go.Figure()

                    if n_classes == 2:
                        prob = y_score[:, 1] if y_score.ndim > 1 else y_score
                        fpr, tpr, _ = roc_curve(y_test, prob)
                        roc_auc = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                            name=f"AUC = {roc_auc:.4f}", line=dict(width=2)))
                    else:
                        y_bin = label_binarize(y_test, classes=classes)
                        for i, cls in enumerate(classes):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                            roc_auc = auc(fpr, tpr)
                            lbl = le_target.classes_[cls] if le_target else str(cls)
                            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                name=f"{lbl} (AUC={roc_auc:.3f})", line=dict(width=2)))

                    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                      line=dict(dash="dash", color="gray"))
                    fig_roc.update_layout(
                        template="plotly_white",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        title="ROC Curve",
                        height=chart_height,
                        legend=dict(orientation="h")
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as roc_err:
                st.warning(f"ROC non disponibile: {roc_err}")

            # Feature Importance
            if model_name == "Random Forest":
                imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
                st.plotly_chart(px.bar(imp, orientation="h", title="Feature Importance",
                                       height=chart_height), use_container_width=True)

            code = f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

df = pd.read_csv("your_file.csv").dropna()
X = df[{feature_cols}]
y = df["{target_col}"]
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
if y.dtype == object:
    y = LabelEncoder().fit_transform(y.astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
model = {model.__class__.__name__}()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred, average="weighted"))"""

        else:
            mse  = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MSE", f"{mse:.4f}")
            m2.metric("RMSE", f"{rmse:.4f}")
            m3.metric("MAE", f"{mae:.4f}")
            m4.metric("R² Score", f"{r2:.4f}")

            if model_name in ["Linear Regression", "Ridge Regression"]:
                st.subheader("📐 Coefficienti del modello")
                
                intercept = model.intercept_
                coef_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficiente": model.coef_
                }).sort_values("Coefficiente", key=abs, ascending=False)
                
                col_i, col_c = st.columns(2)
                col_i.metric("Intercetta (bias)", f"{intercept:.6f}")
                
                st.dataframe(coef_df, use_container_width=True, hide_index=True)
                
                # Grafico coefficienti
                fig_coef = px.bar(
                    coef_df, x="Coefficiente", y="Feature",
                    orientation="h",
                    title="Coefficienti del modello",
                    color="Coefficiente",
                    color_continuous_scale="RdBu",
                    height=chart_height
                )
                fig_coef.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_coef.update_layout(template="plotly_white")
                st.plotly_chart(fig_coef, use_container_width=True)


            residuals = np.array(y_test) - np.array(y_pred)
            fig_res = px.scatter(x=y_pred, y=residuals,
                                 labels={"x": "Predicted", "y": "Residuals"},
                                 title="Residual Plot", height=chart_height)
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)

            mean_res = float(np.mean(residuals))
            std_res  = float(np.std(residuals))
            x_range  = np.linspace(mean_res - 4*std_res, mean_res + 4*std_res, 200)
            y_norm   = stats.norm.pdf(x_range, mean_res, std_res)
            fig_dist = px.histogram(x=residuals, nbins=30, histnorm="probability density",
                title=f"Distribuzione residui — μ={mean_res:.4f}, σ={std_res:.4f}",
                labels={"x": "Residuo", "y": "Densità"}, height=chart_height)
            fig_dist.add_scatter(x=x_range, y=y_norm, mode="lines",
                                 name="Curva normale", line=dict(width=2))
            fig_dist.add_vline(x=mean_res, line_dash="dash",
                               annotation_text=f"μ={mean_res:.4f}", annotation_position="top right")
            fig_dist.update_layout(template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

            if model_name == "Random Forest":
                imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
                st.plotly_chart(px.bar(imp, orientation="h", title="Feature Importance",
                                       height=chart_height), use_container_width=True)

            code = f"""import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("your_file.csv").dropna()
X = df[{feature_cols}]
y = df["{target_col}"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
model = {model.__class__.__name__}()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))"""

        st.divider()
        st.subheader("⬇️ Esporta")
        buf = io.BytesIO()
        joblib.dump(model, buf)
        st.download_button("📥 Scarica modello (.pkl)", data=buf.getvalue(),
                           file_name=f"{model_name.replace(' ', '_')}.pkl",
                           mime="application/octet-stream")
        st.subheader("🐍 Codice Python equivalente")
        st.code(code, language="python")

    except Exception as e:
        st.error(f"Errore: {e}")