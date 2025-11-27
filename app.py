from flask import Flask, render_template, jsonify
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import base64
import io
import sqlite3
import os
import datetime

app = Flask(__name__)

# ==========================================
# 1) GARANTE QUE O BANCO EXISTE SEMPRE
# ==========================================

def criar_tabela():
    conn = sqlite3.connect("previsoes.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_prevista TEXT,
            valor_previsto REAL,
            criado_em TEXT
        )
    """)
    conn.commit()
    conn.close()

# 🔥 SE O BANCO SUMIR NO AZURE, CRIA DE NOVO
if not os.path.exists("previsoes.db"):
    criar_tabela()


# ==========================================
# 2) CARREGAR DADOS
# ==========================================

df = pd.read_csv("14_Petroleo_INT.csv")
df["data"] = pd.to_datetime(df["data"])
df.set_index("data", inplace=True)

# Treina o modelo ARIMA
modelo = auto_arima(df["valor"], seasonal=False)


# ==========================================
# 3) GERAR PREVISÕES E SALVAR NO BANCO
# ==========================================

periodos = 6
forecast = modelo.predict(n_periods=periodos)

datas_futuras = pd.date_range(df.index[-1], periods=periodos + 1, closed="right")
previsoes = list(zip(datas_futuras.strftime("%Y-%m-%d"), forecast))

def salvar_previsoes(previsoes_list):
    conn = sqlite3.connect("previsoes.db")
    cur = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for data_p, valor in previsoes_list:
        cur.execute("""
            INSERT INTO previsoes (data_prevista, valor_previsto, criado_em)
            VALUES (?, ?, ?)
        """, (data_p, float(valor), agora))
    conn.commit()
    conn.close()

salvar_previsoes(previsoes)


# ==========================================
# 4) LEITURA DO BANCO PARA EXIBIR NA PÁGINA
# ==========================================

def obter_previsoes():
    conn = sqlite3.connect("previsoes.db")
    cur = conn.cursor()
    cur.execute("SELECT data_prevista, valor_previsto FROM previsoes ORDER BY data_prevista")
    dados = cur.fetchall()
    conn.close()
    return dados


# ==========================================
# 5) ROTAS DO SITE
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/modelo")
def modelo_page():
    previsoes_bd = obter_previsoes()
    return render_template("modelo.html", previsoes=previsoes_bd)


@app.route("/grafico")
def grafico():

    fig, ax = plt.subplots(figsize=(12, 5))

    # Série histórica
    ax.plot(df.index, df["valor"], color="white", label="Histórico")

    # Forecast
    ax.plot(datas_futuras, forecast, marker="o", color="lime", label="Forecast")

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return render_template("grafico.html", grafico=img_base64)


# ==========================================
# 6) API OPCIONAL
# ==========================================

@app.route("/api/previsoes")
def api_previsoes():
    return jsonify({"previsoes": obter_previsoes()})


# ==========================================
# 7) INICIAR LOCALMENTE
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
