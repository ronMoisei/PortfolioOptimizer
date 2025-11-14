# model/selector.py
from __future__ import annotations

from typing import Dict, Sequence, Mapping, Any, List, Tuple
from collections import Counter
import numpy as np
import pandas as pd


class PortfolioSelector:
    """
    Classe OOP per eseguire la selezione combinatoria (Step 4 & 5).

    - Contiene:
        * dati: rho, rating_scores, sectors, has_rating, mu
        * parametri: K, rating_min, max_unrated_share, alpha, beta, gamma, delta, ecc.
    - Il metodo .select(candidati) lancia:
        * un seed greedy iniziale
        * una DFS con branch-and-bound usando un bound approssimato (rating+mu).
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

        # 2. Parametri / Vincoli
        self.K = int(params.get("K", 0))
        self.rating_min = float(params.get("rating_min", 0.0))
        self.max_unrated_share = float(params.get("max_unrated_share", 1.0))
        self.max_share_per_sector = float(params.get("max_share_per_sector", 1.0))
        self.max_count_per_sector_param = params.get("max_count_per_sector", None)
        self.rho_pair_max = params.get("rho_pair_max", None)

        # 3. Pesi dello Score
        self.alpha = float(params.get("alpha", 1.0))
        self.beta = float(params.get("beta", 0.0))
        self.gamma = float(params.get("gamma", 0.0))
        self.delta = float(params.get("delta", 0.0))

        # 4. Stato risultati
        self.best_subset: List[str] | None = None
        self.best_score: float = float("-inf")

        # 5. Vincolo K per settore
        self.max_count_per_sector: int | None = None
        if self.max_count_per_sector_param is not None:
            self.max_count_per_sector = int(self.max_count_per_sector_param)
        elif self.max_share_per_sector < 1.0 and self.K > 0:
            self.max_count_per_sector = max(
                1, int(np.floor(self.max_share_per_sector * self.K + 1e-9))
            )

        # 6. Strutture per B&B
        self._candidati_ordered: List[str] = []
        self._potential: List[float] = []
        self._prefix_potential: np.ndarray | None = None

    @staticmethod
    def build_default_params() -> Dict[str, Any]:
        """
        Costruisce un dizionario di parametri di default per il selettore.
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

    # ==========================
    #  STEP 4: SCORE E UTILITY
    # ==========================

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
        """ Penalità settoriale, usando self.sectors e self.max_share_per_sector """
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
        """ Score completo sul subset (usato solo sui leaf) """
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

        score = (
            self.alpha * (-mean_corr)
            + self.beta * mean_rating
            - self.gamma * pen_sector
            + self.delta * mean_return
        )
        return float(score)

    # ==========================
    #  VINCOLI HARD
    # ==========================

    def _violates_constraints(self, parziale: Sequence[str], new_t: str) -> bool:
        """ Verifica i vincoli hard usando i parametri pre-calcolati in self """
        new_subset = list(parziale) + [new_t]
        n_total = len(new_subset)

        # 1) Rating minimo
        if self.has_rating.get(new_t, False):
            rs = self.rating_scores.get(new_t)
            if rs is not None and rs < self.rating_min:
                return True

        # 2) Limiti per settore
        if self.max_count_per_sector is not None:
            sec_counts = Counter(
                self.sectors.get(t) for t in new_subset if self.sectors.get(t) is not None
            )
            sec_new = self.sectors.get(new_t)
            if sec_new is not None and sec_counts.get(sec_new, 0) > self.max_count_per_sector:
                return True

        # 3) Quota max unrated
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

    # ==========================
    #  GREEDY SEED
    # ==========================

    def _greedy_seed(self, candidati: Sequence[str]) -> List[str]:
        """
        Costruisce un portafoglio greedy:
        - parte da insieme vuoto
        - ad ogni passo aggiunge il titolo che massimizza lo score completo
          (tra quelli che non violano i vincoli hard).
        """
        if self.K <= 0:
            return []

        chosen: List[str] = []

        for _ in range(self.K):
            best_t = None
            best_s = float("-inf")

            for t in candidati:
                if t in chosen:
                    continue
                if self._violates_constraints(chosen, t):
                    continue

                trial_subset = chosen + [t]
                s = self._getScore(trial_subset)
                if s > best_s:
                    best_s = s
                    best_t = t

            if best_t is None:
                break

            chosen.append(best_t)

        return chosen

    # ==========================
    #  POTENZIALE PER-ASSET (BOUND)
    # ==========================

    def _asset_potential(self, t: str) -> float:
        """
        Potenziale "ottimistico" del singolo asset, basato SOLO su rating e mu:

            pot(t) = max( beta * rating_t + delta * mu_t, 0 )

        Parte di correlazione e penalità settoriale vengono ignorate → bound più ottimistico.
        """
        rs = self.rating_scores.get(t)
        hr = self.has_rating.get(t, False)
        rating_term = self.beta * float(rs) if (hr and rs is not None) else 0.0

        mu_val = float(self.mu.get(t, 0.0))
        return_term = self.delta * mu_val

        pot = rating_term + return_term
        if pot < 0.0:
            pot = 0.0
        return pot

    # ==========================
    #  DFS + BRANCH-AND-BOUND
    # ==========================

    def _ricorsione(
        self,
        parziale: List[str],
        start_idx: int,
        sum_rating: float,
        count_rating: int,
        sum_mu: float,
        count_mu: int,
    ) -> None:
        """
        Motore di ricorsione con branch-and-bound.

        Qui usiamo un bound approssimato:
        - approx_partial = beta * avg_rating(parziale) + delta * avg_mu(parziale)
        - max_extra = somma dei migliori 'remaining_to_pick' potenziali
                      nella coda [start_idx:], usando i prefix-sum.
        """
        # Base case
        if len(parziale) == self.K:
            s = self._getScore(parziale)  # valutazione completa
            if s > self.best_score:
                self.best_score = s
                self.best_subset = list(parziale)
            return

        remaining_to_pick = self.K - len(parziale)
        if remaining_to_pick <= 0:
            return

        n = len(self._candidati_ordered)
        remaining_total = n - start_idx
        if remaining_total < remaining_to_pick:
            return

        # Score approssimato del subset corrente (rating + mu)
        if count_mu > 0 or count_rating > 0:
            avg_rating = (sum_rating / count_rating) if count_rating > 0 else 0.0
            avg_mu = (sum_mu / count_mu) if count_mu > 0 else 0.0
            approx_partial = self.beta * avg_rating + self.delta * avg_mu
        else:
            approx_partial = 0.0

        # Max extra potenziale dalla coda [start_idx:]
        # (candidati già ordinati per potenziale decrescente)
        j = start_idx + remaining_to_pick
        if j > n:
            j = n
        max_extra = float(self._prefix_potential[j] - self._prefix_potential[start_idx])

        optimistic_bound = approx_partial + max_extra

        # Potatura: se neanche nel caso migliore supereremmo best_score
        if optimistic_bound <= self.best_score:
            return

        # DFS: prova ad aggiungere ciascun candidato rimanente
        for i in range(start_idx, n):
            t = self._candidati_ordered[i]
            if self._violates_constraints(parziale, t):
                continue

            # aggiorna accumulatori
            new_sum_rating = sum_rating
            new_count_rating = count_rating
            if self.has_rating.get(t, False):
                rs = self.rating_scores.get(t)
                if rs is not None:
                    new_sum_rating += float(rs)
                    new_count_rating += 1

            mu_val = float(self.mu.get(t, 0.0))
            new_sum_mu = sum_mu + mu_val
            new_count_mu = count_mu + 1

            parziale.append(t)
            self._ricorsione(
                parziale,
                i + 1,
                new_sum_rating,
                new_count_rating,
                new_sum_mu,
                new_count_mu,
            )
            parziale.pop()

    # ==========================
    #  METODO PUBBLICO
    # ==========================

    def select(self, candidati: Sequence[str]) -> Tuple[List[str] | None, float]:
        """
        Funzione di ingresso per il selettore combinatorio.

        - Riordina i candidati per potenziale per-asset (rating+mu).
        - Pre-calcola prefix-sum dei potenziali.
        - Usa un seed greedy per inizializzare best_score.
        - Esegue DFS con branch-and-bound usando un bound basato solo su rating+mu.
        """
        # 1) Calcola potenziale per ogni candidato e ordina (decrescente)
        data: List[Tuple[str, float]] = []
        for t in candidati:
            pot = self._asset_potential(t)
            data.append((t, pot))

        data.sort(key=lambda x: x[1], reverse=True)

        self._candidati_ordered = [t for t, _ in data]
        self._potential = [p for _, p in data]

        n = len(self._potential)
        self._prefix_potential = np.zeros(n + 1, dtype=float)
        for i, p in enumerate(self._potential, start=1):
            self._prefix_potential[i] = self._prefix_potential[i - 1] + p

        # 2) Reset stato best
        self.best_subset = None
        self.best_score = float("-inf")

        # 3) Seed greedy (score completo, una sola volta)
        seed = self._greedy_seed(self._candidati_ordered)
        if len(seed) == self.K:
            self.best_subset = list(seed)
            self.best_score = self._getScore(seed)

        # 4) DFS + B&B con bound approssimato (rating+mu)
        self._ricorsione(
            parziale=[],
            start_idx=0,
            sum_rating=0.0,
            count_rating=0,
            sum_mu=0.0,
            count_mu=0,
        )

        return self.best_subset, self.best_score


# ==========================
#  TEST HARNESS
# ==========================

if __name__ == "__main__":
    import traceback

    print("=== TEST SELECTOR (OOP + B&B ottimizzato) ===")

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

        params = PortfolioSelector.build_default_params()
        params["K"] = 3
        params["max_unrated_share"] = 0.34
        params["rating_min"] = 13.0
        params["max_share_per_sector"] = 0.67

        selector = PortfolioSelector(
            rho=rho_test,
            rating_scores=rating_scores,
            sectors=sectors,
            has_rating=has_rating,
            mu=mu,
            params=params,
        )

        best_subset, best_score = selector.select(candidati=tickers)

        print("Best subset:", best_subset)
        print("Best score:", best_score)

    except Exception as e:
        print("TEST SELECTOR FALLITO:")
        traceback.print_exc()
    else:
        print("TEST SELECTOR OK.")
