# model/portfolio_weights.py
from __future__ import annotations

from typing import Dict, Sequence, Literal, Optional
import numpy as np
import pandas as pd


class PortfolioWeights:
    """
    Funzioni per il calcolo dei pesi di portafoglio.

    Modalità:
        - "eq": pesi uguali
        - "mv": mean-variance (Markowitz) solo long con vincolo hard w_i >= 1/(2K)
    """


    @staticmethod
    def _equal_weights(tickers: Sequence[str]) -> Dict[str, float]:
        """
        Assegna pesi uguali a tutti i ticker del portafoglio.
        """
        tickers = list(tickers)
        n = len(tickers)
        if n == 0:
            return {}
        w = 1.0 / n
        return {t: float(w) for t in tickers}

    @staticmethod
    def _round_weights_to_2_decimals(
        tickers: Sequence[str],
        w: np.ndarray,
    ) -> Dict[str, float]:
        """
        Arrotonda i pesi alla seconda cifra decimale cercando di mantenere
        la somma a 1.0 (entro gli inevitabili limiti di arrotondamento).

        NOTA: il vincolo w_i >= 1/(2K) deve essere già rispettato nel vettore w.
        """
        tickers = list(tickers)
        w = np.asarray(w, dtype=float)

        # normalizza a somma 1 (per sicurezza)
        s = w.sum()
        if s <= 0:
            n = len(tickers)
            if n == 0:
                return {}
            w = np.ones(n) / n
        else:
            w = w / s

        # Arrotonda a 2 decimali
        w_rounded = np.round(w, 2)

        # Aggiusta eventuale differenza sulla somma
        diff = 1.0 - w_rounded.sum()
        if abs(diff) > 1e-6 and len(w_rounded) > 0:
            j = int(np.argmax(w_rounded))
            w_rounded[j] += diff

        return {t: float(w_i) for t, w_i in zip(tickers, w_rounded)}


    @staticmethod
    def _mean_variance_weights(
        tickers: Sequence[str],
        mu: Optional[pd.Series],
        Sigma_sh: Optional[pd.DataFrame],
        risk_aversion: float = 1.0,
        ridge: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Calcola pesi mean-variance (Markowitz) per i titoli in 'tickers',
        usando mu e Sigma_sh.

        Vincoli:
            - solo long: w_i >= 0
            - vincolo hard: w_i >= 1/(2K) con K = len(tickers)
            - somma w_i = 1

        Se qualcosa va storto (matrice singolare, mu/Sigma mancanti, ecc.),
        fallback ai pesi uguali.
        """
        tickers = list(tickers)
        n = len(tickers)
        if n == 0:
            return {}

        # Se mancano mu o Sigma_sh, fallback
        if mu is None or Sigma_sh is None:
            return PortfolioWeights._equal_weights(tickers)

        # Controllo che tutti i tickers siano presenti in mu e Sigma_sh
        missing_mu = [t for t in tickers if t not in mu.index]
        missing_Sigma = [t for t in tickers if t not in Sigma_sh.index or t not in Sigma_sh.columns]
        if missing_mu or missing_Sigma:
            return PortfolioWeights._equal_weights(tickers)

        # Estrai vettore mu e sottomatrice di covarianza
        mu_vec = mu.loc[tickers].astype(float).values  # shape (n,)
        Sigma_sub = Sigma_sh.loc[tickers, tickers].astype(float).values  # shape (n, n)

        # Regolarizzazione
        Sigma_reg = Sigma_sub + ridge * np.eye(n)
        ones = np.ones(n)

        try:
            A = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            # Matrice non invertibile -> fallback
            return PortfolioWeights._equal_weights(tickers)

        # Soluzione mean-variance con vincolo somma w_i = 1
        c = risk_aversion * mu_vec
        Ac = A @ c
        A1 = A @ ones
        num = ones @ Ac - 1.0
        den = ones @ A1

        if abs(den) < 1e-12:
            return PortfolioWeights._equal_weights(tickers)

        lam = num / den
        w_raw = A @ (c - lam * ones)  # soluzione unconstrained sul simplex

        # -----  vincolo w_i >= 1/(2K) -----

        # 1) Impedisci valori negativi
        w_raw = np.maximum(w_raw, 0.0)

        # Se tutti sono zero, fallback
        if w_raw.sum() <= 0:
            return PortfolioWeights._equal_weights(tickers)

        # 2) Floor w_min = 1/(2K)
        w_min = 1.0 / (2.0 * n)  # vincolo hard desiderato

        # Somma minima dovuta ai floor
        floor_sum = n * w_min
        if floor_sum >= 1.0:
            # Caso patologico: floor troppo alto -> fallback pesi uguali
            return PortfolioWeights._equal_weights(tickers)

        # 3) Residuo da distribuire sopra il floor
        residual = 1.0 - floor_sum
        if residual <= 0:
            return PortfolioWeights._equal_weights(tickers)

        # 4) Distribuisco il residual proporzionalmente ai pesi raw
        base = w_raw.copy()
        base_sum = base.sum()
        if base_sum <= 0:
            return PortfolioWeights._equal_weights(tickers)

        base = base / base_sum        # somma 1
        z = residual * base           # parte "libera"

        w_final = w_min + z           # ogni w_i >= w_min, somma a 1

        # 5) Arrotonda a 2 decimali
        return PortfolioWeights._round_weights_to_2_decimals(tickers, w_final)


    @staticmethod
    def compute(
        tickers: Sequence[str],
        mode: Literal["eq", "mv"] = "mv",
        mu: Optional[pd.Series] = None,
        Sigma_sh: Optional[pd.DataFrame] = None,
        risk_aversion: float = 1.0,
    ) -> Dict[str, float]:
        """
        Entry point unico per il calcolo dei pesi.

        Parametri:
            - tickers: lista dei titoli nel portafoglio ottimo (lunghezza K)
            - mode: "eq" (equal weight) o "mv" (mean-variance)
            - mu: serie dei rendimenti attesi
            - Sigma_sh: matrice di covarianza shrinkata
            - risk_aversion: parametro di rischio per mean-variance
        """
        tickers = list(tickers)

        if mode == "eq" or mu is None or Sigma_sh is None:
            return PortfolioWeights._equal_weights(tickers)

        return PortfolioWeights._mean_variance_weights(
            tickers=tickers,
            mu=mu,
            Sigma_sh=Sigma_sh,
            risk_aversion=risk_aversion,
        )
