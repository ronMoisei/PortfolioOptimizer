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

    # ---------- PRE-PROCESSING RENDIMENTI ----------

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
    def winsorize(
        returns_df: pd.DataFrame,
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> pd.DataFrame:
        """
        Applica una winsorization per colonne tra i quantili 'lower' e 'upper'.

        Esempio tipico: lower=0.01, upper=0.99.
        """
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(
                "lower e upper devono essere quantili in [0, 1] con lower < upper"
            )

        wins = returns_df.copy()
        # quantili per colonna
        qs = wins.quantile([lower, upper], axis=0)
        lower_q = qs.loc[lower]
        upper_q = qs.loc[upper]

        wins = wins.clip(lower=lower_q, upper=upper_q, axis=1)
        return wins

    # ---------- STIMA DI BASE ----------

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

        La finestra temporale da usare (sub-DataFrame) viene passata già estratta
        come 'returns_window'.
        """
        if returns_window is None or returns_window.empty:
            raise ValueError("returns_window è vuoto.")

        # rimuovo righe completamente NaN
        R = returns_window.dropna(how="all").copy()
        T_len = len(R)
        if T_len == 0:
            raise ValueError(
                "returns_window è vuoto dopo aver rimosso le righe tutte NaN."
            )

        # filtro colonne con abbastanza osservazioni non-NaN
        valid_counts = R.notna().sum(axis=0)
        min_non_na = int(np.ceil(min_non_na_ratio * T_len))
        valid_cols = valid_counts[valid_counts >= min_non_na].index.tolist()

        R = R[valid_cols]

        if R.shape[1] < 2:
            raise ValueError(
                "Numero di titoli valido < 2, impossibile stimare ρ e Σ."
            )

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
            raise ValueError(
                "shrink_target non supportato. Usa 'diagonal' o 'identity'."
            )

        # shrinkage semplice: convex combination tra Sigma e Tmat
        Sigma_sh = (1.0 - shrink_lambda) * Sigma + shrink_lambda * Tmat

        return rho, Sigma_sh, mu


    @staticmethod
    def estimate_from_returns(
        returns_df: pd.DataFrame,
        shrink_lambda: float = 0.1,
        min_non_na_ratio: float = 0.8,
        winsor_lower: float | None = 0.01,
        winsor_upper: float | None = 0.99,
        shrink_target: str = "diagonal",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Stima rho, Sigma_sh, mu usando TUTTA la storia contenuta in returns_df.

        Passi:
        - valida che returns_df non sia vuoto,
        - applica eventualmente winsorization,
        - chiama estimate_corr_cov_mu sul DataFrame risultante.
        """
        if returns_df is None or returns_df.empty:
            raise ValueError("returns_df è vuoto.")

        R = returns_df.copy()

        if winsor_lower is not None and winsor_upper is not None:
            R = RiskEstimator.winsorize(
                R,
                lower=winsor_lower,
                upper=winsor_upper,
            )

        return RiskEstimator.estimate_corr_cov_mu(
            R,
            shrink_lambda=shrink_lambda,
            min_non_na_ratio=min_non_na_ratio,
            shrink_target=shrink_target,
        )


if __name__ == "__main__":
    # Test rapido indipendente con dati fittizi

    print("=== TEST RISK_ESTIMATOR (OOP) ===")

    # Simuliamo 500 giorni di prezzi per 4 titoli
    dates = pd.date_range(start="2018-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    prices_data = np.exp(
        np.cumsum(
            rng.normal(loc=0.0005, scale=0.02, size=(len(dates), 4)),
            axis=0,
        )
    ) * 100.0

    prices_df = pd.DataFrame(
        prices_data,
        index=dates,
        columns=["A", "B", "C", "D"],
    )

    # 1) rendimenti
    returns_df = RiskEstimator.compute_returns(prices_df)
    print("Shape returns_df:", returns_df.shape)

    try:
        # 2) stima completa su tutta la storia
        rho, Sigma_sh, mu = RiskEstimator.estimate_from_returns(
            returns_df,
            shrink_lambda=0.1,
            min_non_na_ratio=0.8,
            winsor_lower=0.01,
            winsor_upper=0.99,
        )

        print("Shape rho:", rho.shape)
        print("Shape Sigma_sh:", Sigma_sh.shape)
        print("Shape mu:", mu.shape)

        if rho.shape[0] != rho.shape[1]:
            raise AssertionError("rho non è quadrata.")
        if Sigma_sh.shape != rho.shape:
            raise AssertionError("Sigma_sh ha shape diversa da rho.")
        if list(mu.index) != list(rho.columns):
            raise AssertionError(
                "L'indice di mu non coincide con le colonne di rho."
            )

    except AssertionError as e:
        print("TEST RISK_ESTIMATOR FALLITO:")
        print(" -", e)

    except Exception as e:
        print("ERRORE IMPREVISTO DURANTE IL TEST RISK_ESTIMATOR:")
        print(" -", repr(e))

    else:
        print("TEST RISK_ESTIMATOR OK.")
