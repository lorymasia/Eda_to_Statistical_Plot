# 📊 Eda to Chart & ML Models

A multi-page web app built with **Streamlit** that lets you load a dataset, explore and preprocess it, generate interactive charts with **Plotly**, and train **Machine Learning** models — all without writing a single line of code.

## Pages

### 📊 CSV to Chart
- Load a dataset and generate interactive charts
- 7 chart types: Line, Bar, Scatter, Heatmap, Histogram, Box, Area
- Bar chart with groupby aggregation (mean, sum, median, min, max) or simple count
- Select X and Y axes from available columns
- Optional color grouping by category
- Export charts as **PNG** or interactive **HTML**
- Automatic generation of the equivalent **Python code**

### 🛠️ Data Preprocessing
- Full dataset exploration: statistics, null values, distributions, correlation heatmap
- Column data types overview (type, unique values, example)
- Data cleaning: remove duplicates, handle null values (drop, fill with mean/median/mode/custom), drop columns, rename columns
- Transformations: Label Encoding, One-Hot Encoding, StandardScaler / MinMaxScaler / RobustScaler, Log Transform, Binning, column type casting
- Export the preprocessed dataset as CSV
- Automatic generation of the full **Python preprocessing pipeline**

### 🤖 Supervised Learning
- Auto-detection of task type: **classification** or **regression**
- Classification models: Random Forest, Logistic Regression, SVM, KNN
- Regression models: Random Forest, Linear Regression, Ridge, SVR
- Metrics — Classification: Accuracy, F1, Precision, Recall + Confusion Matrix
- Metrics — Regression: MSE, RMSE, MAE, R² + Residual Plot + Residuals distribution with normal curve
- Feature Importance chart (Random Forest)
- Export trained model as **.pkl**
- Automatic generation of the equivalent **Python code**

### 🔍 Unsupervised Learning
- Clustering algorithms: **K-Means** and **DBSCAN**
- Elbow Method visualization for optimal K selection
- Silhouette Score evaluation
- 2D visualization via **PCA** or **t-SNE**
- Cluster statistics per group
- Export dataset with cluster labels as CSV
- Automatic generation of the equivalent **Python code**

## Dataset Sources
All pages support three ways to load a dataset:
- 📂 **Upload CSV** from your computer
- 📦 **ISLP library** built-in datasets (Auto, Boston, Wage, etc.)
- 🔗 **Direct URL** (e.g. raw GitHub or public CSV links)

The loaded dataset persists across all pages — no need to reload when switching.

## Requirements

- Python 3.8+
- streamlit
- pandas
- plotly
- kaleido
- scikit-learn
- joblib
- numpy
- scipy
- requests
- ISLP *(optional)*

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

## Project Structure

```
├── app.py                  # 📊 CSV to Chart
├── pages/
│   ├── 1_Preprocessing.py  # 🛠️ Data Preprocessing
│   ├── 2_Supervised.py     # 🤖 Supervised Learning
│   └── 3_Unsupervised.py   # 🔍 Unsupervised Learning
├── style.css
├── requirements.txt
├── .gitignore
└── README.md
```
