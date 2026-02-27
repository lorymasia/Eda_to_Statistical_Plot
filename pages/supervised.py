import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error)
import joblib
import requests
import io

def load_css(filepath):
    with open(filepath, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


st.set_page_config(page_title="Supervised Learning", layout="wide", page_icon="🤖")
st.title("🤖 Supervised Learning")
st.markdown("Carica un dataset, scegli le feature, il target e il modello da addestrare.")

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

# DOPO (corretto)
all_cols = df.columns.tolist()
col1, col2 = st.columns(2)
feature_cols = col1.multiselect(
    "📥 Feature (X) — seleziona più colonne",
    all_cols,
    default=all_cols[:-1]
)
target_col = col2.selectbox(
    "🎯 Target (Y) — una sola colonna",
    [c for c in all_cols if c not in feature_cols],
    index=0
)


if not feature_cols:
    st.warning("Seleziona almeno una feature.")
    st.stop()

task_type = "classification" if df[target_col].dtype == object or df[target_col].nunique() <= 10 else "regression"
st.info(f"**Task rilevato:** {'📊 Classificazione' if task_type == 'classification' else '📈 Regressione'}")

col3, col4 = st.columns(2)
if task_type == "classification":
    model_name = col3.selectbox("Modello", ["Random Forest", "Logistic Regression", "SVM", "KNN"])
else:
    model_name = col3.selectbox("Modello", ["Random Forest", "Linear Regression", "Ridge Regression", "SVR"])

test_size = col4.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
normalize = st.checkbox("Normalizza feature (StandardScaler)", value=True)

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
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(random_state=42),
            "KNN": KNeighborsClassifier()
        }
        models_reg = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "SVR": SVR()
        }

        model = models_clf[model_name] if task_type == "classification" else models_reg[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📊 Risultati")

        if task_type == "classification":
            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("F1 (weighted)", f"{f1:.4f}")
            m3.metric("Precision (weighted)", f"{prec:.4f}")
            m4.metric("Recall (weighted)", f"{rec:.4f}")

            cm = confusion_matrix(y_test, y_pred)
            labels = le_target.classes_.tolist() if le_target else list(map(str, sorted(set(y_test))))
            fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                               color_continuous_scale="Blues", title="Confusion Matrix",
                               labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm, use_container_width=True)

            if model_name == "Random Forest":
                imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
                st.plotly_chart(px.bar(imp, orientation="h", title="Feature Importance"), use_container_width=True)

            code = f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
print("F1:", f1_score(y_test, y_pred, average="weighted"))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))"""

        else:
            mse  = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MSE",      f"{mse:.4f}")
            m2.metric("RMSE",     f"{rmse:.4f}")
            m3.metric("MAE",      f"{mae:.4f}")
            m4.metric("R² Score", f"{r2:.4f}")

            residuals = np.array(y_test) - np.array(y_pred)
            fig_res = px.scatter(x=y_pred, y=residuals,
                                 labels={"x": "Predicted", "y": "Residuals"}, title="Residual Plot")
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)

            mean_res = float(np.mean(residuals))
            std_res  = float(np.std(residuals))
            x_range  = np.linspace(mean_res - 4*std_res, mean_res + 4*std_res, 200)
            y_norm   = stats.norm.pdf(x_range, mean_res, std_res)
            fig_dist = px.histogram(
                x=residuals, nbins=30, histnorm="probability density",
                title=f"Distribuzione residui  —  μ = {mean_res:.4f},  σ = {std_res:.4f}",
                labels={"x": "Residuo", "y": "Densità"}
            )
            fig_dist.add_scatter(x=x_range, y=y_norm, mode="lines",
                                 name="Curva normale", line=dict(width=2))
            fig_dist.add_vline(x=mean_res, line_dash="dash",
                               annotation_text=f"μ = {mean_res:.4f}",
                               annotation_position="top right")
            fig_dist.update_layout(template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

            if model_name == "Random Forest":
                imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
                st.plotly_chart(px.bar(imp, orientation="h", title="Feature Importance"), use_container_width=True)

            if model_name == "Random Forest":
                imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
                st.plotly_chart(px.bar(imp, orientation="h", title="Feature Importance"), use_container_width=True)

            code = f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

df = pd.read_csv("your_file.csv").dropna()
X = df[{feature_cols}]
y = df["{target_col}"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
model = {model.__class__.__name__}()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:",  mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:",  mean_absolute_error(y_test, y_pred))
print("R2:",   r2_score(y_test, y_pred))"""

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