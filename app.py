from flask import Flask, jsonify, send_file, render_template
import pandas as pd
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io
import sqlite3
import datetime
import os

app = Flask(__name__)

# =============================
# CONFIGURA BANCO DE DADOS
# =============================

DB_PATH = "database.db"

def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE previsoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                modelo TEXT,
                data_execucao TEXT,
                valor_previsto REAL
            )
        """)
        conn.commit()
        conn.close()

init_db()

def salvar_previsao(modelo, valor):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO previsoes (modelo, data_execucao, valor_previsto) VALUES (?, ?, ?)",
                (modelo, str(datetime.datetime.now()), float(valor)))
    conn.commit()
    conn.close()


# =============================
# CARREGAR DADOS
# =============================

df = pd.read_csv("14_Petroleo_INT.csv", index_col=0, parse_dates=True)
df = df.rename(columns={"DCOILWTICO": "petroleo"})

# Modelo ARIMA automático
model = auto_arima(df.petroleo, seasonal=False, trace=False)
forecast = model.predict(n_periods=6)

# salvar no banco a última previsão
salvar_previsao("ARIMA automático", forecast[-1])


# =============================
# ROTAS HTML (SITE)
# =============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/modelo")
def pagina_modelo():
    ultimo_valor = forecast[-1]
    return render_template("modelo.html", previsao=round(ultimo_valor, 2))


# =============================
# ROTAS DE API
# =============================

@app.route("/status")
def status():
    return {"status": "API de séries temporais funcionando!"}


@app.route("/dados")
def dados():
    return df.to_json()


@app.route("/forecast")
def previsao():
    return jsonify(forecast.tolist())


@app.route("/grafico")
def grafico():
    buf = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df.petroleo)
    plt.title("Preço do Petróleo")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# =============================
# EXECUTAR LOCALMENTE
# =============================

if __name__ == "__main__":
    app.run(debug=True)
