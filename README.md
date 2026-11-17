# 📈 Previsão de Preços de Ações da Amazon (AMZN)

**Projeto Integrador IV — Ciência de Dados**  
Aplicação de modelos de Machine Learning, Deep Learning e técnicas avançadas de **Ensemble** para previsão do preço de fechamento (`Close`) das ações da Amazon (`AMZN`) em dois horizontes temporais: **curto prazo ($t+1$)** e **longo prazo ($t+90$)**.

---

## 🎯 Objetivo do Projeto

Desenvolver, avaliar e colocar em produção um pipeline preditivo livre de vazamento de dados (*data leakage*) seguindo a metodologia **CRISP-DM**, comparando abordagens lineares regularizadas, redes neurais recorrentes, máquinas de vetores de suporte e métodos de combinação (*Ensemble*).

### Horizontes de Previsão
* **Horizonte Imediato ($t+1$):** Previsão do preço de fechamento do próximo pregão.
* **Horizonte Longo ($t+90$):** Previsão do preço de fechamento 90 pregões à frente.

---

## 🧠 Metodologia e Arquitetura (CRISP-DM)

1. **Coleta e Compreensão dos Dados:**
   * Histórico diário da `AMZN` desde `1997-05-15` (via Kaggle / `yfinance`), complementado por variáveis macroeconômicas e de mercado (`DXY`, `^VIX`, `^IXIC`, `MSFT`, `GOOGL`, `AAPL`, `^TNX`) na etapa exploratória.
2. **Preparação e Prevenção de *Data Leakage*:**
   * Remoção de `Close(t)` das variáveis explicativas para evitar colinearidade trivial.
   * Uso estrito das variáveis conhecidas no instante $t$: `Open`, `High`, `Low`, `Volume` e `Retornos` (retorno logarítmico).
   * Divisão temporal sequencial (80% treino / 20% teste) com aplicação de *gap* temporal (`Data_alvo < Data_teste_min`) e validação cruzada via `TimeSeriesSplit`.
3. **Modelagem Individual & Deep Learning:**
   * **Ridge Regression** ($\alpha = 1.0$) e **Lasso Regression** ($\alpha = 0.01$).
   * **Support Vector Regression (SVR)** com kernel linear ($C = 1$, $\epsilon = 0.01$).
   * **LSTM Univariado** (janelas temporais de 30 dias / 10 semanas).
4. **Validação Estatística e Métodos de Ensemble:**
   * **Moving Block Bootstrap:** 1.000 réplicas com blocos de 15 pregões para avaliar a estabilidade estatística das métricas.
   * **Voting Regressor:** Combinação por média das previsões de Ridge, Lasso e SVR.
   * **Blending & Stacking CV:** Meta-modelo `LinearRegression` treinado sobre previsões *Out-Of-Fold* (`TimeSeriesSplit`).
   * **Bagging Regressor:** Reamostragem bootstrap com pipelines padronizados.
   * **Boosting Residual (Customizado):** Arquitetura aditiva sequencial onde cada estimador aprende e corrige o erro residual deixado pela etapa anterior.

---

## 📊 Resumo dos Resultados Experimentais

### Comparativo: Holdout Temporal vs. Moving Block Bootstrap (1.000 réplicas)

| Horizonte | Modelo | RMSE (Holdout) | $R^2$ (Holdout) | RMSE Médio (Bootstrap) | $R^2$ Médio (Bootstrap) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$t+1$ (1 dia)** | **Ridge** | 3.28 | 0.9919 | **1.52** | **0.9993** |
| **$t+1$ (1 dia)** | **Lasso** | 3.27 | 0.9920 | 2.01 | 0.9988 |
| **$t+1$ (1 dia)** | **SVR** | 3.50 | 0.9955 | 11.94 | 0.9555 |
| **$t+90$ (90 dias)** | **Ridge** | 26.12 | 0.4801 | 11.89 | 0.9560 |
| **$t+90$ (90 dias)** | **Lasso** | 26.17 | 0.4784 | **11.85** | **0.9564** |
| **$t+90$ (90 dias)** | **SVR** | **22.39** | **0.8108** | 11.95 | 0.9557 |

### Principais Conclusões
* **Curto Prazo ($t+1$):** Modelos lineares regularizados (**Ridge** e **Lasso**) e o **Voting Regressor** apresentam altíssima precisão ($R^2 > 0.99$) e baixo erro absoluto.
* **Longo Prazo ($t+90$):** A incerteza acumulada degrada modelos lineares simples no holdout ($R^2 \approx 0.48$), enquanto o **SVR** e os métodos de **Ensemble** (**Boosting Residual** e **Stacking**) mantêm maior capacidade de generalização e controle de variância.

---

## 🗂️ Estrutura do Repositório

```text
├── models/
│   └── artefatos_amzn.joblib                # Modelos treinados, scalers e métricas serializados
├── ProjetoIntegrador_IV_GitHub_Completo.ipynb # Notebook completo com EDA, modelos e ensembles
├── train.py                                 # Script de treino, validação e exportação dos artefatos
├── app.py                                   # Aplicação web interativa em Streamlit
├── requirements.txt                         # Dependências do projeto
└── README.md                                # Documentação do repositório
```

---

## 🚀 Como Executar Localmente

### 1. Clonar o repositório e instalar as dependências
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
pip install -r requirements.txt
```

### 2. Treinar e gerar os artefatos dos modelos
```bash
python train.py
```
*(Este comando baixa o histórico atualizado da `AMZN` via Yahoo Finance, executa a validação temporal e salva os modelos em `models/artefatos_amzn.joblib`.)*

### 3. Iniciar o dashboard Streamlit
```bash
streamlit run app.py
```

---

## ☁️ Deploy no Streamlit Cloud

1. Suba os arquivos `app.py`, `train.py`, `requirements.txt` e `models/artefatos_amzn.joblib` para o seu repositório no GitHub.
2. Acesse [share.streamlit.io](https://share.streamlit.io/) e conecte sua conta do GitHub.
3. Clique em **New app**, selecione o repositório, defina `app.py` como arquivo principal (*Main file path*) e clique em **Deploy**.

---

## ⚠️ Aviso Legal

> Este projeto tem caráter estritamente acadêmico e experimental, desenvolvido no âmbito do **Projeto Integrador IV**. As previsões geradas pelos modelos não constituem recomendação de investimento ou aconselhamento financeiro.