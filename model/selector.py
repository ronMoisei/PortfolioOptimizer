# model/selector.py
from __future__ import annotations

from typing import Dict, Sequence, Mapping, Any, List, Tuple
from collections import Counter
import numpy as np
import pandas as pd


class PortfolioSelector:
    """
    Classe per eseguire la selezione combinatoria.

    Questa classe viene istanziata con tutti i dati (rho, mu, ratings) e
    i parametri (K, alpha, beta, ...).

    Questo design evita di passare 10+ parametri a ogni chiamata ricorsiva.
    """

    def __init__(
            self,
            rho: pd.DataFrame,
            rating_scores: Mapping[str, float | None],
            sectors: Mapping[str, str | None],
            has_rating: Mapping[str, bool],
            mu: Mapping[str, float],
            params: Mapping[str, Any],
    ):
        # 1. Dati
        self.rho = rho
        self.rating_scores = rating_scores
        self.sectors = sectors
        self.has_rating = has_rating
        self.mu = mu
        self.params = params

        # 2. Parametri / Vincoli (pre-calcolati per efficienza)
        self.K = int(params.get("K", 0))
        self.rating_min = float(params.get("rating_min", 0.0))
        self.max_unrated_share = float(params.get("max_unrated_share", 1.0))
        self.max_share_per_sector = float(params.get("max_share_per_sector", 1.0))
        self.max_count_per_sector_param = params.get("max_count_per_sector", None)
        self.rho_pair_max = params.get("rho_pair_max", None)

        # 3. Pesi dello Score (pre-calcolati per efficienza)
        self.alpha = float(params.get("alpha", 1.0))
        self.beta = float(params.get("beta", 0.0))
        self.gamma = float(params.get("gamma", 0.0))
        self.delta = float(params.get("delta", 0.0))

        # 4. Stato dei risultati
        self.best_subset: List[str] | None = None
        self.best_score: float = float("-inf")

        # 5. Vincolo K per settore (calcolato una sola volta)
        self.max_count_per_sector: int | None = None
        if self.max_count_per_sector_param is not None:
            self.max_count_per_sector = int(self.max_count_per_sector_param)
        elif self.max_share_per_sector < 1.0 and self.K > 0:
            self.max_count_per_sector = max(1, int(np.floor(self.max_share_per_sector * self.K + 1e-9)))

    @staticmethod
    def build_default_params() -> Dict[str, Any]:
        """
        Costruisce un dizionario di parametri di default per il selettore.
        (Statico perché non dipende da nessuna istanza)
        """
        params = {
            "K": 20,
            "rating_min": 13.0,
            "max_share_per_sector": 0.4,
            "rho_pair_max": None,
            "max_unrated_share": 0.2,
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 0.5,
            "delta": 0.2,
        }
        return params

    # ---------- METODI PRIVATI ----------

    def _avg_corr(self, subset: Sequence[str], use_abs: bool = True) -> float:
        """ Calcola la correlazione media, usando self.rho """
        n = len(subset)
        if n < 2:
            return 0.0

        sub_rho = self.rho.loc[subset, subset]
        mat = sub_rho.values
        iu = np.triu_indices(n, k=1)
        vals = mat[iu]
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            return 0.0
        if use_abs:
            vals = np.abs(vals)

        return float(vals.mean())

    def _sector_penalty(self, subset: Sequence[str]) -> float:
        """ Calcola la penalità settoriale, usando self.sectors e self.max_share_per_sector """
        n = len(subset)
        if n == 0:
            return 0.0

        max_share = self.max_share_per_sector
        if max_share <= 0.0:
            max_share = 0.0

        counts: Dict[str, int] = {}
        for t in subset:
            sec = self.sectors.get(t)
            if sec is None:
                continue
            counts[sec] = counts.get(sec, 0) + 1

        penalty = 0.0
        for sec, c in counts.items():
            share = c / n
            excess = share - max_share
            if excess > 0:
                penalty += excess

        return float(penalty)

    def _getScore(self, subset: Sequence[str]) -> float:
        """ Calcola lo score, usando i pesi e i dati in self """
        if len(subset) == 0:
            return float("-inf")

        # 1) Correlazione
        mean_corr = self._avg_corr(subset, use_abs=True)

        # 2) Rating
        rated_vals: List[float] = []
        for t in subset:
            if self.has_rating.get(t, False):
                rs = self.rating_scores.get(t)
                if rs is not None:
                    rated_vals.append(float(rs))
        mean_rating = float(np.mean(rated_vals)) if rated_vals else 0.0

        # 3) Penalità settoriale
        pen_sector = self._sector_penalty(subset)

        # 4) Rendimento
        returns_vals: List[float] = []
        for t in subset:
            if t in self.mu and self.mu[t] is not None:
                returns_vals.append(float(self.mu[t]))
        mean_return = float(np.mean(returns_vals)) if returns_vals else 0.0

        # Calcolo finale
        score = (
                self.alpha * (-mean_corr)
                + self.beta * mean_rating
                - self.gamma * pen_sector
                + self.delta * mean_return
        )
        return float(score)

    # ---------- RICORSIONE ----------

    def _violates_constraints(self, parziale: Sequence[str], new_t: str) -> bool:
        """ Verifica i vincoli hard usando i parametri pre-calcolati in self """

        new_subset = list(parziale) + [new_t]
        n_total = len(new_subset)

        # 1) Rating minimo
        if self.has_rating.get(new_t, False):
            rs = self.rating_scores.get(new_t)
            if rs is not None and rs < self.rating_min:
                return True

        # 2) Limiti per settore (usa self.max_count_per_sector calcolato in __init__)
        if self.max_count_per_sector is not None:
            sec_counts = Counter(self.sectors.get(t) for t in new_subset if self.sectors.get(t) is not None)
            sec_new = self.sectors.get(new_t)
            if sec_new is not None and sec_counts.get(sec_new, 0) > self.max_count_per_sector:
                return True

        # 3) Quota massima unrated
        if self.max_unrated_share < 1.0 and n_total > 0:
            n_unrated = sum(1 for t in new_subset if not self.has_rating.get(t, False))
            share_unrated = n_unrated / n_total
            if share_unrated > self.max_unrated_share:
                return True

        # 4) Max correlazione per coppia
        if self.rho_pair_max is not None:
            rho_max = float(self.rho_pair_max)
            for t in parziale:
                try:
                    cij = self.rho.loc[t, new_t]
                except KeyError:
                    continue
                if pd.notna(cij) and abs(cij) > rho_max:
                    return True

        return False

    def _ricorsione(
            self,
            parziale: List[str],
            start_idx: int,
            candidati: Sequence[str],
    ) -> None:

        # Base case: portafoglio completo
        if len(parziale) == self.K:
            s = self._getScore(parziale)
            if s > self.best_score:
                self.best_score = s
                self.best_subset = list(parziale)
            return

        remaining = len(candidati) - start_idx
        if len(parziale) + remaining < self.K:
            return

        # Loop ricorsivo
        for i in range(start_idx, len(candidati)):
            t = candidati[i]

            # Controllo vincoli hard
            if self._violates_constraints(parziale, t):
                continue

            parziale.append(t)
            self._ricorsione(parziale, i + 1, candidati)
            parziale.pop()  # Backtrack

    # ---------- METODI PUBBLICI ----------

    def select(self, candidati: Sequence[str]) -> Tuple[List[str] | None, float]:
        """
        Funzione di ingresso per il selettore combinatorio.
        Avvia la ricerca e restituisce i risultati.
        """
        # Resetta lo stato per una nuova run
        self.best_subset = None
        self.best_score = float("-inf")

        # Avvia la ricorsione
        self._ricorsione([], 0, candidati)

        return self.best_subset, self.best_score


if __name__ == "__main__":
    import traceback

    print("=== TEST SELECTOR (OOP) ===")

    try:
        tickers = ["A", "B", "C", "D", "E"]
        data = {
            "A": [1.0, 0.6, 0.1, -0.2, 0.3],
            "B": [0.6, 1.0, 0.5, 0.0, 0.2],
            "C": [0.1, 0.5, 1.0, 0.3, 0.4],
            "D": [-0.2, 0.0, 0.3, 1.0, 0.1],
            "E": [0.3, 0.2, 0.4, 0.1, 1.0],
        }
        rho_test = pd.DataFrame(data, index=tickers, columns=tickers)
        rating_scores = {"A": 18.0, "B": 16.0, "C": 13.0, "D": None, "E": 10.0}
        has_rating = {t: (rating_scores[t] is not None) for t in tickers}
        sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Energy", "E": "Finance"}
        mu = {"A": 0.08, "B": 0.07, "C": 0.06, "D": 0.12, "E": 0.09}

        # --- MODIFICA CHIAVE ---
        # 1. Costruisci i parametri (usando il metodo statico)
        params = PortfolioSelector.build_default_params()
        params["K"] = 3
        params["max_unrated_share"] = 0.34
        params["rating_min"] = 13.0
        params["max_share_per_sector"] = 0.67

        # 2. Istanzia il Selettore
        selector = PortfolioSelector(
            rho=rho_test,
            rating_scores=rating_scores,
            sectors=sectors,
            has_rating=has_rating,
            mu=mu,
            params=params,
        )

        # 3. Esegui la selezione
        best_subset, best_score = selector.select(candidati=tickers)
        # ---------------------

        print("Best subset:", best_subset)
        print("Best score:", best_score)

    except Exception as e:
        print("TEST SELECTOR FALLITO:")
        traceback.print_exc()
    else:
        print("TEST SELECTOR OK.")