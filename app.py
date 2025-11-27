from flask import Flask, render_template, Markup
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from pmdarima import auto_arima
import numpy as np
import datetime

app = Flask(__name__)

# ---------- Helpers ----------
def load_csv(path="14_Petroleo_INT.csv"):
    """
    Carrega o CSV e tenta detectar as colunas de data/valor automaticamente.
    Espera ter uma coluna de data (index ou nome 'data', 'Date', 'DATE')
    e uma coluna de valor (ex: 'valor', 'value', 'DCOILWTICO', 'price').
    """
    df = pd.read_csv(path)
    # tentar detectar coluna de data
    date_cols = [c for c in df.columns if c.lower() in ("data","date","dt","timestamp")]
    if not date_cols and df.columns.size > 0 and df.columns[0].lower() not in ("valor","value","dcoilwtico","price"):
        # fallback: se primeira coluna parece uma data (strings com - or /), tentar parse
        date_cols = [df.columns[0]]
    if date_cols:
        df.rename(columns={date_cols[0]: "data"}, inplace=True)
        df["data"] = pd.to_datetime(df["data"])
        df.set_index("data", inplace=True)
    else:
        # se já vem indexado, tentar forçar parse do index
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # detectar coluna de valor
    val_candidates = [c for c in df.columns if c.lower() in ("valor","value","dcoilwtico","price","preco","preço")]
    if val_candidates:
        val_col = val_candidates[0]
    else:
        # fallback para a primeira coluna numérica
        numeric = df.select_dtypes(include=[float,int]).columns
        if len(numeric) > 0:
            val_col = numeric[0]
        else:
            # última tentativa: segunda coluna
            val_col = df.columns[0]
    df = df[[val_col]].rename(columns={val_col: "valor"})
    df = df.sort_index()
    return df

def plot_series_to_base64(df):
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df["valor"], label="Histórico", color="#00d4ff")
    ax.set_title("Preço do Barril de Petróleo — Série Histórica")
    ax.set_xlabel("Data")
    ax.set_ylabel("USD por barril")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def forecast_table_html(df, periods=6):
    # treina modelo auto_arima (com supressões para evitar logs demais)
    modelo = auto_arima(df["valor"], seasonal=False, error_action="ignore", suppress_warnings=True)
    forecast, conf_int = modelo.predict(n_periods=periods, return_conf_int=True)

    last_date = df.index.max()
    # gera datas começando no próximo mês
    futuras = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq="M")

    # montar tabela HTML simples (retornável para template)
    rows = []
    for d, f, ci in zip(futuras, forecast, conf_int):
        rows.append({
            "data": d.strftime("%Y-%m-%d"),
            "previsao": float(f),
            "low": float(ci[0]),
            "high": float(ci[1])
        })

    # transforma em HTML via pandas para ficar bonito
    df_out = pd.DataFrame(rows)
    df_out["previsao"] = df_out["previsao"].map(lambda x: f"US$ {x:,.2f}")
    df_out["intervalo"] = df_out.apply(lambda r: f"US$ {r['low']:,.2f} — US$ {r['high']:,.2f}", axis=1)
    df_out = df_out[["data","previsao","intervalo"]]
    html_table = df_out.to_html(index=False, classes="forecast-table", border=0, justify="center", escape=False)
    return html_table

# ---------- Carregar dados no startup para performance ----------
try:
    df = load_csv("14_Petroleo_INT.csv")
except Exception as e:
    # dataframe vazio de fallback
    df = pd.DataFrame({"valor":[]})

# pré-gerar o PNG da série histórica
try:
    series_img_b64 = plot_series_to_base64(df)
except Exception:
    series_img_b64 = ""

# ---------- ROTAS ----------

@app.route("/")
def index():
    # Texto principal: explicação curta + motivos
    descricao = """
    <p>Este trabalho analisa o <b>preço do barril de petróleo (em USD)</b> ao longo do tempo e tenta prever
    os valores dos próximos meses. A informação sobre o preço do petróleo é relevante para economia, transporte,
    combustíveis e custo de produção industrial.</p>
    <p><b>Por que isso importa?</b></p>
    <ul>
      <li><b>Combustíveis:</b> o preço do barril influencia diretamente o preço da gasolina e diesel, afetando o custo de deslocamento e transporte de mercadorias.</li>
      <li><b>Inflação:</b> aumento no preço do petróleo pode elevar custos de transporte e produção, pressionando os preços ao consumidor.</li>
      <li><b>Empregos e renda:</b> economias dependentes de energia sentem impacto em empregos e receitas públicas.</li>
      <li><b>Orçamento doméstico:</b> famílias percebem aumento em combustíveis, gás e produtos transportados, reduzindo renda disponível.</li>
    </ul>
    """
    return render_template("index.html", descricao=Markup(descricao))

@app.route("/grafico")
def grafico():
    # mostra o gráfico histórico do CSV e botão para previsões
    return render_template("grafico.html", img_series=series_img_b64)

@app.route("/forecast")
def forecast():
    # gera a tabela do forecast (6 meses) e mostra
    try:
        table_html = forecast_table_html(df, periods=6)
    except Exception as e:
        table_html = f"<p>Erro ao gerar forecast: {str(e)}</p>"
    return render_template("forecast.html", table_html=Markup(table_html))

# rota simples para checar se está vivo
@app.route("/status")
def status():
    return {"status":"ok"}

if __name__ == "__main__":
    app.run(debug=True)
