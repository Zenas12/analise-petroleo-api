from flask import Flask, render_template
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import base64, io, datetime

from database import criar_tabela, salvar_previsoes, obter_previsoes

app = Flask(__name__)

# Criar tabela ao iniciar
criar_tabela()

# Carregar dataset
df = pd.read_csv("14_Petroleo_INT.csv", index_col=0, parse_dates=True)
df = df.rename(columns={"DCOILWTICO": "petroleo"})

# Treinar modelo
modelo = auto_arima(df.petroleo, seasonal=False, trace=False)

# Gerar previsão dos próximos 6 meses
periodos = 6
forecast = modelo.predict(n_periods=periodos)

# Criar datas futuras
datas_futuras = pd.date_range(df.index[-1], periods=periodos + 1, closed="right")

# Listar pares (data, valor)
previsoes = list(zip(datas_futuras.strftime("%Y-%m-%d"), forecast))

# 🔥 Salvar previsões no banco
salvar_previsoes(previsoes)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/modelo")
def modelo_page():
    previsoes_bd = obter_previsoes()
    return render_template("modelo.html", previsoes=previsoes_bd)


@app.route("/grafico")
def grafico():

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df.index, df.petroleo, label="Histórico", color="white")
    ax.plot(datas_futuras, forecast, label="Forecast", color="lime")

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return render_template("grafico.html", grafico=img_b64)
