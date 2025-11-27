from flask import Flask, render_template
from markupsafe import Markup   # CORRETO PARA FLASK 3.0
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from pmdarima import auto_arima
import numpy as np
import datetime


app = Flask(__name__)


# ----------------------------------------------------------
#  HELPERS
# ----------------------------------------------------------

def load_csv(path="14_Petroleo_INT.csv"):
    df = pd.read_csv(path)

    # detectar coluna de data
    date_cols = [c for c in df.columns if c.lower() in ("data", "date", "dt", "timestamp")]
    if not date_cols:
        date_cols = [df.columns[0]]  # fallback

    df.rename(columns={date_cols[0]: "data"}, inplace=True)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df.set_index("data", inplace=True)

    # detectar coluna valor
    val_cols = [c for c in df.columns if c.lower() in ("valor", "value", "dcoilwtico", "price", "preco", "preço")]
    if val_cols:
        val = val_cols[0]
    else:
        val = df.select_dtypes(include=[float, int]).columns[0]

    df = df[[val]].rename(columns={val: "valor"})
    df = df.sort_index()

    return df


def plot_series_to_base64(df):
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["valor"], color="#00d4ff", label="Histórico")
    ax.set_title("Preço do Petróleo — Série Histórica")
    ax.set_xlabel("Data")
    ax.set_ylabel("USD por barril")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def forecast_table_html(df, periods=6):

    modelo = auto_arima(df["valor"], seasonal=False, error_action="ignore", suppress_warnings=True)
    forecast, ci = modelo.predict(n_periods=periods, return_conf_int=True)

    last = df.index.max()
    datas = pd.date_range(start=last + pd.DateOffset(months=1), periods=periods, freq="M")

    linhas = []
    for d, f, c in zip(datas, forecast, ci):
        linhas.append({
            "data": d.strftime("%Y-%m-%d"),
            "previsao": f"US$ {f:,.2f}",
            "intervalo": f"US$ {c[0]:,.2f} — US$ {c[1]:,.2f}"
        })

    df_out = pd.DataFrame(linhas)
    return df_out.to_html(index=False, classes="forecast-table", border=0)


# ----------------------------------------------------------
#  CARREGAR AO INICIAR A APLICAÇÃO
# ----------------------------------------------------------

try:
    df = load_csv("14_Petroleo_INT.csv")
except:
    df = pd.DataFrame({"valor": []})

try:
    series_img_b64 = plot_series_to_base64(df)
except:
    series_img_b64 = ""


# ----------------------------------------------------------
#  ROTAS
# ----------------------------------------------------------

@app.route("/")
def index():

    descricao = Markup("""
    <p>Este trabalho analisa o <b>preço do barril de petróleo (USD)</b> e prevê os próximos meses.</p>
    <p><b>Por que isso importa?</b></p>
    <ul>
      <li><b>Combustíveis</b> – gasolina e diesel dependem do preço do petróleo.</li>
      <li><b>Inflação</b> – aumenta custos de transporte e produção.</li>
      <li><b>Empregos</b> – indústrias dependem do preço da energia.</li>
      <li><b>Orçamento familiar</b> – afeta gás, combustível e transporte.</li>
    </ul>
    """)

    return render_template("index.html", descricao=descricao)


@app.route("/grafico")
def grafico():
    return render_template("grafico.html", img_series=series_img_b64)


@app.route("/forecast")
def forecast():
    try:
        tabela = Markup(forecast_table_html(df))
    except Exception as e:
        tabela = f"<p>Erro ao gerar previsão: {e}</p>"
    return render_template("forecast.html", table_html=tabela)


@app.route("/status")
def status():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
