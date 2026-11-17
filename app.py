import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from train import BoostingResidual, prever_stacking  # Necessário para desserializar o joblib

st.set_page_config(
    page_title="Previsão de Ações AMZN | Projeto Integrador V",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def carregar_artefatos():
    return joblib.load("models/artefatos_amzn.joblib")


@st.cache_data(ttl=3600)
def buscar_dados_recentes():
    df = yf.download("AMZN", period="1y", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Retornos"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna().reset_index(drop=True)


def gerar_previsao(modelo_obj, nome_modelo, X_input):
    if nome_modelo == "Stacking CV":
        return float(prever_stacking(modelo_obj, X_input)[0])
    return float(modelo_obj.predict(X_input)[0])


def gerar_previsoes_lote(modelo_obj, nome_modelo, X_df):
    if nome_modelo == "Stacking CV":
        return prever_stacking(modelo_obj, X_df)
    return modelo_obj.predict(X_df)


artefatos = carregar_artefatos()
df_recente = buscar_dados_recentes()
features = artefatos["features"]

# --- BARRA LATERAL ---
st.sidebar.header("Configurações de Previsão")
horizonte = st.sidebar.radio(
    "Horizonte Temporal:",
    options=[1, 90],
    format_func=lambda x: "t+1 (Próximo Pregão)" if x == 1 else "t+90 (90 Pregões à Frente)",
)

info_h = artefatos["horizontes"][horizonte]
df_metricas = info_h["metricas"]
melhor_modelo_padrao = df_metricas.iloc[0]["Modelo"]

modelo_selecionado = st.sidebar.selectbox(
    "Modelo Preditivo:",
    options=df_metricas["Modelo"].tolist(),
    index=0,
    help="Ordenado automaticamente pelo menor RMSE no conjunto de teste temporal.",
)

usar_simulacao = st.sidebar.checkbox("Simular cenário manual (What-If)", value=False)

ultimo_pregao = df_recente.iloc[-1]
data_ref_dt = ultimo_pregao["Date"]
data_ref = data_ref_dt.strftime("%d/%m/%Y")
data_alvo_dt = data_ref_dt + pd.tseries.offsets.BDay(horizonte)
data_alvo_str = data_alvo_dt.strftime("%d/%m/%Y")
close_atual = float(ultimo_pregao["Close"])

if usar_simulacao:
    st.sidebar.subheader("Parâmetros Simulados")
    in_open = st.sidebar.number_input("Open ($)", value=float(ultimo_pregao["Open"]), step=0.5)
    in_high = st.sidebar.number_input("High ($)", value=float(ultimo_pregao["High"]), step=0.5)
    in_low = st.sidebar.number_input("Low ($)", value=float(ultimo_pregao["Low"]), step=0.5)
    in_vol = st.sidebar.number_input("Volume", value=float(ultimo_pregao["Volume"]), step=1_000_000.0)
    in_ret = st.sidebar.number_input("Retorno Logarítmico", value=float(ultimo_pregao["Retornos"]), format="%.5f")
    X_entrada = pd.DataFrame([[in_open, in_high, in_low, in_vol, in_ret]], columns=features)
else:
    X_entrada = df_recente[features].iloc[[-1]]

# --- CORPO PRINCIPAL ---
st.title("Previsão de Preços de Ações — Amazon (AMZN)")
st.markdown(
    f"**Projeto Integrador V** • Base: Pregão de **{data_ref}** → Alvo: **{data_alvo_str}** (`t+{horizonte}`) "
    f"• 🏆 Melhor modelo histórico: **{melhor_modelo_padrao}**"
)

valor_previsto = gerar_previsao(info_h["modelos"][modelo_selecionado], modelo_selecionado, X_entrada)
variacao_pct = ((valor_previsto - close_atual) / close_atual) * 100
metrica_atual = df_metricas[df_metricas["Modelo"] == modelo_selecionado].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Fechamento Base ({data_ref})", f"US$ {close_atual:.2f}")
col2.metric(
    f"Previsão p/ {data_alvo_str} ({modelo_selecionado})",
    f"US$ {valor_previsto:.2f}",
    f"{variacao_pct:+.2f}%",
)
col3.metric("RMSE (Teste Temporal)", f"US$ {metrica_atual['RMSE']:.2f}")
col4.metric("Coeficiente R² (Teste)", f"{metrica_atual['R2']:.4f}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Histórico & Projeção",
    "Comparativo de Modelos & Ensembles",
    "Validação Temporal (Out-of-Sample)",
    "Diário de Previsões (Previsto vs. Real)",
])

with tab1:
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=df_recente["Date"],
            y=df_recente["Close"],
            mode="lines",
            name="Fechamento Real (AMZN)",
            line=dict(color="#FF9900", width=2.2),
        )
    )

    fig_hist.add_trace(
        go.Scatter(
            x=[data_ref_dt, data_alvo_dt],
            y=[close_atual, valor_previsto],
            mode="lines+markers",
            name=f"Projeção {modelo_selecionado} ({data_alvo_str})",
            line=dict(color="#00CC96", width=2.5, dash="dash"),
            marker=dict(size=10),
        )
    )

    fig_hist.update_layout(
        title=f"Histórico Recente AMZN (12 meses) e Projeção para {data_alvo_str} (t+{horizonte})",
        xaxis_title="Data",
        yaxis_title="Preço de Fechamento (USD)",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader(f"Comparativo de Modelos — Base: {data_ref} | Alvo: {data_alvo_str} (t+{horizonte})")
    if usar_simulacao:
        st.caption("⚠️ Exibindo previsões calculadas com os **parâmetros manuais (What-If)** da barra lateral.")

    previsoes_todos = []
    for _, row in df_metricas.iterrows():
        nome_m = row["Modelo"]
        prev_m = gerar_previsao(info_h["modelos"][nome_m], nome_m, X_entrada)
        previsoes_todos.append({
            "Modelo": nome_m,
            "Data Base (Fechamento)": "Simulação Manual" if usar_simulacao else data_ref,
            "Fechamento Base (US$)": round(close_atual, 2),
            "Data Alvo Previsão": data_alvo_str,
            "Previsão (US$)": round(prev_m, 2),
            "Variação (%)": f"{((prev_m - close_atual) / close_atual) * 100:+.2f}%",
            "RMSE Teste (US$)": round(row["RMSE"], 4),
            "MAE Teste (US$)": round(row["MAE"], 4),
            "R² Teste": round(row["R2"], 4),
        })

    st.dataframe(pd.DataFrame(previsoes_todos), use_container_width=True, hide_index=True)

    if modelo_selecionado == "Boosting Residual" or "Boosting Residual" in df_metricas["Modelo"].values:
        st.info(f"**Sequência selecionada via TimeSeriesSplit no Boosting Residual:** `{info_h['sequencia_boosting']}`")

with tab3:
    st.subheader(f"Real vs. Previsto no Período de Teste ({melhor_modelo_padrao})")
    fig_val = go.Figure()
    fig_val.add_trace(
        go.Scatter(
            x=info_h["datas_teste"],
            y=info_h["y_test"],
            mode="lines",
            name="Valor Real",
            line=dict(color="#636EFA", width=1.8),
        )
    )
    fig_val.add_trace(
        go.Scatter(
            x=info_h["datas_teste"],
            y=info_h["y_pred_melhor"],
            mode="lines",
            name=f"Previsto ({melhor_modelo_padrao})",
            line=dict(color="#EF553B", width=1.8, dash="dot"),
        )
    )
    fig_val.update_layout(
        xaxis_title="Data Alvo",
        yaxis_title="Preço de Fechamento (USD)",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_val, use_container_width=True)

with tab4:
    st.subheader(f"Diário de Previsões Recentes — Modelo: {modelo_selecionado} (t+{horizonte})")
    st.markdown(
        "Acompanhamento dia a dia mostrando o **pregão de origem (ontem/base)**, o **valor previsto** "
        "e o **fechamento real** assim que o pregão alvo é encerrado no Yahoo Finance."
    )

    qtd_dias = st.slider("Quantidade de pregões recentes para exibir:", min_value=5, max_value=60, value=15)

    # Garante que haja linhas suficientes para deslocar pelo horizonte + janela exibida
    n_necessario = min(len(df_recente), qtd_dias + horizonte)
    df_log = df_recente.tail(n_necessario).copy().reset_index(drop=True)

    preds_lote = gerar_previsoes_lote(info_h["modelos"][modelo_selecionado], modelo_selecionado, df_log[features])
    df_log["Preco_Previsto"] = preds_lote
    df_log["Data_Alvo_Real"] = df_log["Date"].shift(-horizonte)
    df_log["Preco_Real_Alvo"] = df_log["Close"].shift(-horizonte)

    # Pega apenas os últimos `qtd_dias` pregões (do mais recente para o mais antigo)
    df_exibir = df_log.tail(qtd_dias).iloc[::-1].reset_index(drop=True)

    linhas_diario = []
    for _, r in df_exibir.iterrows():
        dt_base_str = r["Date"].strftime("%d/%m/%Y")
        close_base = float(r["Close"])
        prev_val = float(r["Preco_Previsto"])

        if pd.notna(r["Preco_Real_Alvo"]):
            dt_alvo_str = r["Data_Alvo_Real"].strftime("%d/%m/%Y")
            real_val = float(r["Preco_Real_Alvo"])
            erro_usd = prev_val - real_val
            erro_pct = (abs(erro_usd) / real_val) * 100
            real_fmt = f"US$ {real_val:.2f}"
            erro_fmt = f"US$ {erro_usd:+.2f} ({erro_pct:.2f}%)"
            status = "✅ Realizado"
        else:
            dt_estimada = r["Date"] + pd.tseries.offsets.BDay(horizonte)
            dt_alvo_str = f"{dt_estimada.strftime('%d/%m/%Y')} (est.)"
            real_fmt = "⏳ Aguardando fechamento"
            erro_fmt = "—"
            status = "🕒 Em aberto"

        linhas_diario.append({
            "Status": status,
            "Data Base (Fechamento)": dt_base_str,
            "Fechamento Base (US$)": f"US$ {close_base:.2f}",
            "Data Alvo": dt_alvo_str,
            f"Previsão ({modelo_selecionado})": f"US$ {prev_val:.2f}",
            "Preço Real (Alvo)": real_fmt,
            "Diferença / Erro (%)": erro_fmt,
        })

    df_diario_final = pd.DataFrame(linhas_diario)
    st.dataframe(df_diario_final, use_container_width=True, hide_index=True)

    # Gráfico comparando Previsto vs Real nos pregões já realizados dessa janela
    df_realizados = df_exibir.dropna(subset=["Preco_Real_Alvo"]).iloc[::-1]
    if not df_realizados.empty:
        fig_diario = go.Figure()
        fig_diario.add_trace(
            go.Scatter(
                x=df_realizados["Data_Alvo_Real"],
                y=df_realizados["Preco_Real_Alvo"],
                mode="lines+markers",
                name="Preço Real no Alvo",
                line=dict(color="#FF9900", width=2.2),
            )
        )
        fig_diario.add_trace(
            go.Scatter(
                x=df_realizados["Data_Alvo_Real"],
                y=df_realizados["Preco_Previsto"],
                mode="lines+markers",
                name=f"Previsão ({modelo_selecionado})",
                line=dict(color="#00CC96", width=2.2, dash="dash"),
            )
        )
        fig_diario.update_layout(
            title=f"Acompanhamento Recente: Previsto vs. Real ({modelo_selecionado})",
            xaxis_title="Data Alvo",
            yaxis_title="Preço (USD)",
            hovermode="x unified",
            height=360,
        )
        st.plotly_chart(fig_diario, use_container_width=True)