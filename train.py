from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ["Open", "High", "Low", "Volume", "Retornos"]
MODELOS_BASE = {
    "Lasso": Lasso(alpha=0.01, max_iter=100000, tol=1e-4, random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "SVR": SVR(kernel="linear", C=1, epsilon=0.01, gamma="scale"),
}


class BoostingResidual:
    """Sequência aditiva: cada modelo aprende o resíduo deixado pelos anteriores."""

    def __init__(self, sequencia):
        self.sequencia = tuple(sequencia)

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        y = np.asarray(y, dtype=float)
        previsto = np.zeros(len(y))
        self.modelos_ = []
        for nome, taxa in self.sequencia:
            residuo = y - previsto
            modelo = clone(MODELOS_BASE[nome]).fit(Xs, residuo)
            previsto = previsto + taxa * modelo.predict(Xs)
            self.modelos_.append((nome, taxa, modelo))
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(X)
        previsto = np.zeros(len(X))
        for _, taxa, modelo in self.modelos_:
            previsto = previsto + taxa * modelo.predict(Xs)
        return previsto


def selecionar_sequencia_boosting(treino, horizonte, max_etapas=6):
    cv = TimeSeriesSplit(n_splits=3, gap=horizonte)
    folds = []
    for idx_t, idx_v in cv.split(treino):
        t, v = treino.iloc[idx_t], treino.iloc[idx_v]
        scaler = StandardScaler().fit(t[FEATURES])
        folds.append({
            "X_t": scaler.transform(t[FEATURES]),
            "X_v": scaler.transform(v[FEATURES]),
            "y_t": t["Alvo"].to_numpy(),
            "y_v": v["Alvo"].to_numpy(),
            "pred_t": np.zeros(len(t)),
            "pred_v": np.zeros(len(v)),
        })

    sequencia = []
    melhor_rmse = np.inf
    for etapa in range(1, max_etapas + 1):
        candidatos = []
        for nome, estimador in MODELOS_BASE.items():
            correcoes = []
            for f in folds:
                erro_anterior = f["y_t"] - f["pred_t"]
                modelo = clone(estimador).fit(f["X_t"], erro_anterior)
                correcoes.append((modelo.predict(f["X_t"]), modelo.predict(f["X_v"])))

            for taxa in ((1.0,) if etapa == 1 else (0.1, 0.5, 1.0)):
                scores = [
                    np.sqrt(mean_squared_error(f["y_v"], f["pred_v"] + taxa * c[1]))
                    for f, c in zip(folds, correcoes)
                ]
                candidatos.append((float(np.mean(scores)), nome, taxa, correcoes))

        rmse_atual, nome_sel, taxa_sel, correcoes_sel = min(candidatos, key=lambda x: x[0])
        if etapa > 1 and rmse_atual >= melhor_rmse * 0.999:
            break

        melhor_rmse = rmse_atual
        sequencia.append((nome_sel, taxa_sel))
        for f, c in zip(folds, correcoes_sel):
            f["pred_t"] += taxa_sel * c[0]
            f["pred_v"] += taxa_sel * c[1]

    return sequencia


def treinar_stacking(X_train, y_train, horizonte):
    tscv = TimeSeriesSplit(n_splits=5, gap=horizonte)
    modelos = {
        nome: Pipeline([("scaler", StandardScaler()), ("modelo", clone(est))])
        for nome, est in MODELOS_BASE.items()
    }
    nomes = list(modelos.keys())
    oof = np.full((len(X_train), len(nomes)), np.nan)

    for train_idx, val_idx in tscv.split(X_train):
        X_ft, X_fv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_ft = y_train.iloc[train_idx]
        for i, nome in enumerate(nomes):
            m = clone(modelos[nome]).fit(X_ft, y_ft)
            oof[val_idx, i] = m.predict(X_fv)

    mask = ~np.isnan(oof).any(axis=1)
    meta = LinearRegression().fit(oof[mask], y_train.iloc[mask])

    modelos_finais = {
        nome: clone(modelos[nome]).fit(X_train, y_train) for nome in nomes
    }
    return {"modelos_base": modelos_finais, "meta_modelo": meta, "nomes": nomes}


def prever_stacking(stack_artefato, X):
    preds = [
        stack_artefato["modelos_base"][nome].predict(X)
        for nome in stack_artefato["nomes"]
    ]
    return stack_artefato["meta_modelo"].predict(np.column_stack(preds))


def carregar_dados_treino():
    print("Baixando histórico completo da AMZN via Yahoo Finance...")
    df = yf.download("AMZN", start="1997-05-15", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Retornos"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def main():
    df = carregar_dados_treino()
    artefatos = {"features": FEATURES, "horizontes": {}}

    for horizonte in [1, 90]:
        print(f"\n--- Treinando pipeline para horizonte t+{horizonte} ---")
        base = df.copy()
        base["Alvo"] = base["Close"].shift(-horizonte)
        base["Data_alvo"] = base["Date"].shift(-horizonte)
        dados = base.dropna(subset=FEATURES + ["Alvo", "Data_alvo"]).copy()

        corte = int(len(dados) * 0.8)
        teste = dados.iloc[corte:].copy()
        treino = dados.iloc[:corte].copy()
        # Garante ausência de vazamento temporal (gap)
        treino = treino.loc[treino["Data_alvo"] < teste["Date"].min()].copy()

        X_train, y_train = treino[FEATURES], treino["Alvo"]
        X_test, y_test = teste[FEATURES], teste["Alvo"]

        modelos_treinados = {}
        metricas = []

        # 1. Modelos Individuais (com StandardScaler em Pipeline)
        for nome, est in MODELOS_BASE.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("modelo", clone(est))])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            modelos_treinados[nome] = pipe
            metricas.append({
                "Modelo": nome,
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
                "R2": float(r2_score(y_test, y_pred)),
            })

        # 2. Voting Regressor
        voting = Pipeline([
            ("scaler", StandardScaler()),
            ("voting", VotingRegressor([
                ("Ridge", clone(MODELOS_BASE["Ridge"])),
                ("Lasso", clone(MODELOS_BASE["Lasso"])),
                ("SVR", clone(MODELOS_BASE["SVR"])),
            ])),
        ])
        voting.fit(X_train, y_train)
        y_pred_vot = voting.predict(X_test)
        modelos_treinados["Voting Regressor"] = voting
        metricas.append({
            "Modelo": "Voting Regressor",
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_vot))),
            "MAE": float(mean_absolute_error(y_test, y_pred_vot)),
            "R2": float(r2_score(y_test, y_pred_vot)),
        })

        # 3. Stacking CV
        stack_art = treinar_stacking(X_train, y_train, horizonte)
        y_pred_stk = prever_stacking(stack_art, X_test)
        modelos_treinados["Stacking CV"] = stack_art
        metricas.append({
            "Modelo": "Stacking CV",
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_stk))),
            "MAE": float(mean_absolute_error(y_test, y_pred_stk)),
            "R2": float(r2_score(y_test, y_pred_stk)),
        })

        # 4. Boosting Residual
        seq = selecionar_sequencia_boosting(treino, horizonte)
        boosting = BoostingResidual(seq).fit(X_train, y_train)
        y_pred_bst = boosting.predict(X_test)
        modelos_treinados["Boosting Residual"] = boosting
        metricas.append({
            "Modelo": "Boosting Residual",
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_bst))),
            "MAE": float(mean_absolute_error(y_test, y_pred_bst)),
            "R2": float(r2_score(y_test, y_pred_bst)),
        })

        # Re-treina os modelos com toda a base histórica disponível para inferência futura
        X_full, y_full = dados[FEATURES], dados["Alvo"]
        modelos_producao = {}
        for nome in ["Lasso", "Ridge", "SVR", "Voting Regressor"]:
            modelos_producao[nome] = clone(modelos_treinados[nome]).fit(X_full, y_full)
        modelos_producao["Stacking CV"] = treinar_stacking(X_full, y_full, horizonte)
        modelos_producao["Boosting Residual"] = BoostingResidual(seq).fit(X_full, y_full)

        df_metricas = pd.DataFrame(metricas).sort_values("RMSE").reset_index(drop=True)
        print(df_metricas.to_string(index=False))

        artefatos["horizontes"][horizonte] = {
            "modelos": modelos_producao,
            "metricas": df_metricas,
            "datas_teste": teste["Data_alvo"].dt.strftime("%Y-%m-%d").tolist(),
            "y_test": y_test.tolist(),
            "y_pred_melhor": (
                prever_stacking(modelos_treinados["Stacking CV"], X_test).tolist()
                if df_metricas.iloc[0]["Modelo"] == "Stacking CV"
                else modelos_treinados[df_metricas.iloc[0]["Modelo"]].predict(X_test).tolist()
            ),
            "sequencia_boosting": " -> ".join(f"{n} (x{t:g})" for n, t in seq),
        }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(artefatos, "models/artefatos_amzn.joblib")
    print("\nArtefatos salvos com sucesso em: models/artefatos_amzn.joblib")


if __name__ == "__main__":
    main()