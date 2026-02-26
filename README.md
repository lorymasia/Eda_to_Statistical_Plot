# 📊 CSV to Chart

A simple web app built with **Streamlit** that lets you upload one or more CSV files
and instantly generate interactive charts using **Plotly**.

## Features

- 📂 Upload multiple CSV files and switch between them
- 📈 7 chart types: Line, Bar, Scatter, Heatmap, Histogram, Box, Area
- 🎯 Select X axis, one or multiple Y columns, and an optional color category
- ⬇️ Export charts as **PNG** or interactive **HTML**
- 🐍 Automatic generation of the equivalent **Python code** ready to copy

## Requirements

- Python 3.8+
- streamlit
- pandas
- plotly
- kaleido (for PNG export)

## Installation

```bash
pip install -r requirements.txt
```

## Usage
streamlit run app.py
