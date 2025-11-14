# model/risk_estimator.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class RiskEstimator:
    """
    Classe che raggruppa tutte le funzioni
    relative al pre-processing dei rendimenti e alla stima delle
    matrici di rischio (rho, Sigma) e dei rendimenti attesi (mu).
    """

    @staticmethod
    def compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola i rendimenti log a partire da un DataFrame di prezzi (date x ticker).

        r_{i,t} = log(p_{i,t} / p_{i,t-1})
        """
        prices = prices_df.sort_index().astype(float)
        returns = np.log(prices / prices.shift(1))
        return returns

    @staticmethod
    def winsorize(returns_df: pd.DataFrame,
                  lower: float = 0.01,
                  upper: float = 0.99) -> pd.DataFrame:
        """
        Applica una winsorization per colonne tra i quantili 'lower' e 'upper'.

        Esempio tipico: lower=0.01, upper=0.99.
        """
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("lower e upper devono essere quantili in [0, 1] con lower < upper")

        wins = returns_df.copy()
        # quantili per colonna
        qs = wins.quantile([lower, upper], axis=0)
        lower_q = qs.loc[lower]
        upper_q = qs.loc[upper]

        wins = wins.clip(lower=lower_q, upper=upper_q, axis=1)
        return wins

    @staticmethod
    def estimate_corr_cov_mu(
            returns_window: pd.DataFrame,
            shrink_lambda: float = 0.1,
            min_non_na_ratio: float = 0.8,
            shrink_target: str = "diagonal",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Stima:
        - matrice di correlazione rho
        - matrice di covarianza con shrinkage Sigma_sh
        - vettore dei rendimenti attesi mu
        """
        if returns_window is None or returns_window.empty:
            raise ValueError("returns_window è vuoto.")

        # rimuovo righe completamente NaN
        R = returns_window.dropna(how="all").copy()
        T_len = len(R)
        if T_len == 0:
            raise ValueError("returns_window è vuoto dopo aver rimosso le righe tutte NaN.")

        # filtro colonne con abbastanza osservazioni non-NaN
        valid_counts = R.notna().sum(axis=0)
        min_non_na = int(np.ceil(min_non_na_ratio * T_len))
        valid_cols = valid_counts[valid_counts >= min_non_na].index.tolist()

        R = R[valid_cols]

        if R.shape[1] < 2:
            raise ValueError("Numero di titoli valido < 2, impossibile stimare ρ e Σ.")

        # riempio gli eventuali NaN residui con la media di colonna
        R = R.fillna(R.mean(axis=0))

        # vettore dei rendimenti attesi
        mu = R.mean(axis=0)

        # covarianza e correlazione empiriche
        Sigma = R.cov()
        rho = R.corr()

        if not (0.0 <= shrink_lambda <= 1.0):
            raise ValueError("shrink_lambda deve essere in [0,1].")

        # matrice target T per lo shrinkage
        if shrink_target == "diagonal":
            Tmat = pd.DataFrame(
                np.diag(np.diag(Sigma.values)),
                index=Sigma.index,
                columns=Sigma.columns,
            )
        elif shrink_target == "identity":
            avg_var = float(np.mean(np.diag(Sigma.values)))
            Tmat = pd.DataFrame(
                np.eye(Sigma.shape[0]) * avg_var,
                index=Sigma.index,
                columns=Sigma.columns,
            )
        else:
            raise ValueError("shrink_target non supportato. Usa 'diagonal' o 'identity'.")

        # shrinkage
        Sigma_sh = (1.0 - shrink_lambda) * Sigma + shrink_lambda * Tmat

        return rho, Sigma_sh, mu


if __name__ == "__main__":
    # Test rapido del RiskEstimator usando il Model

    from model.modello import Model

    model = Model()
    try:
        model.load_data_from_dao()

        returns_df = model.returns_df
        if returns_df is None:
            raise AssertionError("returns_df non è stato caricato dal Model.")

        start = returns_df.index[0]
        end = returns_df.index[min(251, len(returns_df) - 1)]

        window = returns_df.loc[start:end]

        print("=== TEST RISK_ESTIMATOR (OOP) ===")
        print(f"Finestra: {start.date()} → {end.date()}")
        print(f"Shape returns_window: {window.shape}")

        rho, Sigma_sh, mu = RiskEstimator.estimate_corr_cov_mu(
            window, shrink_lambda=0.1, min_non_na_ratio=0.8
        )
        # ---------------------

        print(f"Shape rho: {rho.shape}")
        print(f"Shape Sigma_sh: {Sigma_sh.shape}")
        print(f"Shape mu: {mu.shape}")

        if rho.shape[0] != rho.shape[1]:
            raise AssertionError("rho non è quadrata.")
        if Sigma_sh.shape != rho.shape:
            raise AssertionError("Sigma_sh ha shape diversa da rho.")
        if list(mu.index) != list(rho.columns):
            raise AssertionError("L'indice di mu non coincide con le colonne di rho.")

    except AssertionError as e:
        print("TEST RISK_ESTIMATOR FALLITO:")
        print(" -", e)

    except Exception as e:
        print("ERRORE IMPREVISTO DURANTE IL TEST RISK_ESTIMATOR:")
        print(" -", repr(e))

    else:
        print("TEST RISK_ESTIMATOR OK.")