from flask import Flask, send_file, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima
import io

app = Flask(__name__)

# ===============
# CARREGAR DADOS
# ===============
df = pd.read_csv("14_Petroleo_INT.csv", index_col=0, parse_dates=True)
df = df.rename(columns={"DCOILWTICO": "petroleo"})

# Modelo ARIMA automático
model = auto_arima(df.petroleo, seasonal=False, trace=False)
forecast = model.predict(n_periods=6)

# ===============
# ROTAS DA API
# ===============

@app.route("/")
def home():
    return {"status": "API de séries temporais funcionando!"}

@app.route("/dados")
def dados():
    return df.to_json()

@app.route("/forecast")
def forecast_route():
    return jsonify({"forecast_proximos_6_meses": forecast.tolist()})

@app.route("/grafico")
def grafico():
    fig, ax = plt.subplots(figsize=(10, 4))
    df.petroleo.plot(ax=ax, title="Preço do Petróleo — Média Mensal")
    
    # Salvar imagem em memória
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120)
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

# Iniciar app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
