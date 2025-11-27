import sqlite3

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

def salvar_previsoes(lista_previsoes):
    import datetime
    conn = sqlite3.connect("previsoes.db")
    cur = conn.cursor()

    hoje = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for data_p, valor in lista_previsoes:
        cur.execute("""
            INSERT INTO previsoes (data_prevista, valor_previsto, criado_em)
            VALUES (?, ?, ?)
        """, (data_p, float(valor), hoje))

    conn.commit()
    conn.close()

def obter_previsoes():
    conn = sqlite3.connect("previsoes.db")
    cur = conn.cursor()

    cur.execute("SELECT data_prevista, valor_previsto FROM previsoes ORDER BY data_prevista")
    resultados = cur.fetchall()

    conn.close()
    return resultados
