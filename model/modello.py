from __future__ import annotations

from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
import networkx as nx

from database.DAO import DAO
from model.stock import Stock
from model.risk_estimator import RiskEstimator
from model.graph_builder import GraphBuilder
from model.selector import PortfolioSelector


class Model:
    """
    Classe principale del modello.

    Responsabilità:
        - carica dati con il DAO
        - costruisce prices_df, ratings_df, stocks, returns_df
        - stima rho, Sigma_sh, mu su una finestra temporale
        - costruisce il grafo di correlazione
        - applica filtri (soglia, k-NN) e costruisce un universo ridotto U'
        - esegue la selezione combinatoria del portafoglio (Step 5)
        - calcola i pesi (Step 6)
    """

    def __init__(self, dao: Optional[DAO] = None) -> None:
        self._dao = dao if dao is not None else DAO()

        self.prices_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None
        self.stocks: Dict[str, Stock] = {}

        self.returns_df: pd.DataFrame | None = None

        # Oggetti di rischio correnti
        self.current_rho: pd.DataFrame | None = None
        self.current_Sigma_sh: pd.DataFrame | None = None
        self.current_mu: pd.Series | None = None
        self.current_universe: List[str] = []

        # Info su rating
        self.tickers_with_rating: List[str] = []
        self.tickers_without_rating: List[str] = []
        self.has_rating: Dict[str, bool] = {}

        # Grafo e universo ridotto
        self.current_graph: Optional[nx.Graph] = None
        self.reduced_universe: List[str] = []

        # Nuovo: universo usato dal selettore (dopo pre-filtro quantitativo, Step 1)
        self.selector_universe: List[str] = []

        # Risultati ultima ottimizzazione
        self.last_portfolio_tickers: list[str] = []
        self.last_portfolio_weights: Dict[str, float] = {}
        self.last_portfolio_score: float | None = None

    # ---------- STEP 1: CARICAMENTO DATI ----------

    def load_data_from_dao(self) -> None:
        prices, ratings, stock_dict = self._dao.load_universe()

        self.prices_df = prices
        self.ratings_df = ratings
        self.stocks = stock_dict

        # DataFrame dei rendimenti da tutti gli Stock
        returns_dict = {
            ticker: stock.returns
            for ticker, stock in self.stocks.items()
        }
        self.returns_df = pd.DataFrame(returns_dict).sort_index()

        # liste con/senza rating e dizionario has_rating
        self.tickers_with_rating = [
            t for t, s in self.stocks.items() if s.rating_score is not None
        ]
        self.tickers_without_rating = [
            t for t, s in self.stocks.items() if s.rating_score is None
        ]
        self.has_rating = {t: (s.rating_score is not None)
                           for t, s in self.stocks.items()}

    # ---------- STEP 2: STIMA RISCHIO SU FINESTRA TEMPORALE ----------

    def estimate_risk_window(
        self,
        start_date,
        end_date,
        shrink_lambda: float = 0.1,
        min_non_na_ratio: float = 0.8,
        winsor_lower: float | None = 0.01,
        winsor_upper: float | None = 0.99,
    ) -> None:
        if self.returns_df is None:
            raise RuntimeError("returns_df non è stato caricato. Chiama load_data_from_dao() prima.")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        window = self.returns_df.loc[start:end]
        if window.empty:
            raise ValueError("La finestra selezionata non contiene dati.")

        # winsorization opzionale
        if winsor_lower is not None and winsor_upper is not None:
            window = RiskEstimator.winsorize(window, winsor_lower, winsor_upper)

        rho, Sigma_sh, mu = RiskEstimator.estimate_corr_cov_mu(
            window,
            shrink_lambda=shrink_lambda,
            min_non_na_ratio=min_non_na_ratio,
            shrink_target="diagonal",
        )

        self.current_rho = rho
        self.current_Sigma_sh = Sigma_sh
        self.current_mu = mu
        self.current_universe = list(rho.columns)

        # reset grafo/universo ridotto / selector_universe (verranno ricostruiti)
        self.current_graph = None
        self.reduced_universe = []
        self.selector_universe = []

    # ---------- STEP 3: GRAFO E UNIVERSO RIDOTTO ----------

    def build_reduced_universe(
            self,
            tau: float | None = None,
            k: int | None = None,
            max_size: int = 60,
            max_unrated_share: float = 1.0,
            min_rating_score: float = 13.0,
            target_rated_share: float = 0.7,
    ) -> list[str]:
        """
        Costruisce:
        - il grafo di correlazione corrente (self.current_graph),
        - un universo ridotto U' (lista di ticker) salvato in self.reduced_universe.
        """

        if self.current_rho is None:
            raise RuntimeError(
                "current_rho non disponibile. Chiama estimate_risk_window() prima di build_reduced_universe()."
            )

        rho_full = self.current_rho.copy()
        tickers_all = list(rho_full.columns)

        # --- 1) Restrizione iniziale in base a max_unrated_share ---

        if max_unrated_share == 0.0:
            # Universo iniziale: solo titoli con rating (indipendentemente dal valore)
            base_tickers = [t for t in tickers_all if self.has_rating.get(t, False)]
            if not base_tickers:
                raise ValueError("Nessun titolo con rating disponibile nell'universo corrente.")
            rho = rho_full.loc[base_tickers, base_tickers]
        else:
            base_tickers = tickers_all
            rho = rho_full.loc[base_tickers, base_tickers]

        # --- 2) Matrici per filtri di rete (tau, k-NN) ---

        abs_rho = rho.abs().copy()
        np.fill_diagonal(abs_rho.values, 0.0)

        # matrice distanze base
        dist = GraphBuilder.build_distance_matrix(rho, signed=False)

        # filtro a soglia: dove |rho| < tau → distanza = +inf (niente edge)
        if tau is not None:
            if not (0.0 <= tau <= 1.0):
                raise ValueError("tau deve essere in [0,1].")
            mask_below = abs_rho.values < tau
            dist.values[mask_below] = np.inf

        # filtro k-NN (se richiesto)
        if k is not None:
            dist_knn = GraphBuilder.knn_filter(dist, k=k, symmetric=True)
        else:
            dist_knn = dist

        # matrice di adiacenza dei pesi: |rho_ij| se dist_ij finita, 0 altrimenti
        adj = abs_rho.copy()
        mask_no_edge = ~np.isfinite(dist_knn.values)
        adj.values[mask_no_edge] = 0.0
        np.fill_diagonal(adj.values, 0.0)

        # --- 3) Costruzione grafo NetworkX ---

        G = nx.Graph()
        for t in adj.index:
            G.add_node(t)

        for i, ti in enumerate(adj.index):
            for j in range(i + 1, len(adj.columns)):
                tj = adj.columns[j]
                w = adj.iat[i, j]
                if w > 0.0:
                    corr_val = rho.loc[ti, tj]
                    G.add_edge(ti, tj, weight=w, corr=corr_val)

        self.current_graph = G

        # --- 4) Calcolo strength e definizione dei due pool ---

        strength = adj.sum(axis=1)  # somma dei pesi |rho_ij| per ogni nodo
        all_candidates = list(strength.index)

        pool_rated: list[str] = []
        pool_unrated: list[str] = []

        for t in all_candidates:
            stock = self.stocks.get(t)
            rs = stock.rating_score if stock is not None else None
            if rs is not None and rs >= min_rating_score:
                pool_rated.append(t)
            else:
                pool_unrated.append(t)

        # --- 5) Quote per U': N_Rated, N_Unrated ---

        # limito max_size al numero totale di candidati
        if max_size is None or max_size > len(all_candidates):
            max_size = len(all_candidates)

        if max_unrated_share == 0.0:
            # nessun titolo unrated/speculativo nell'universo ridotto:
            target_rated_share_effective = 1.0
        else:
            # di default usiamo target_rated_share, ma garantiamo
            # che la quota target di unrated non superi max_unrated_share
            target_unrated_share = 1.0 - target_rated_share
            if target_unrated_share > max_unrated_share:
                target_unrated_share = max_unrated_share
                target_rated_share_effective = 1.0 - target_unrated_share
            else:
                target_rated_share_effective = target_rated_share

        N_rated_target = int(round(max_size * target_rated_share_effective))
        N_unrated_target = max_size - N_rated_target

        if max_unrated_share == 0.0:
            N_unrated_target = 0
            N_rated_target = max_size

        # clamp ai pool disponibili
        N_rated = min(N_rated_target, len(pool_rated))
        N_unrated = min(N_unrated_target, len(pool_unrated))

        # --- 6) Selezione effettiva per strength bassa ---

        rated_strength = strength.loc[pool_rated].sort_values(ascending=True) if pool_rated else pd.Series(dtype=float)
        unrated_strength = strength.loc[pool_unrated].sort_values(ascending=True) if pool_unrated else pd.Series(dtype=float)

        selected_rated = list(rated_strength.index[:N_rated])
        selected_unrated = list(unrated_strength.index[:N_unrated])

        selected = set(selected_rated + selected_unrated)

        # se mancano ancora titoli per arrivare a max_size, riempio con i restanti (sempre per strength bassa)
        total_selected = len(selected)
        if total_selected < max_size:
            remaining_slots = max_size - total_selected

            remaining_candidates = [
                t for t in all_candidates if t not in selected
            ]
            remaining_strength = strength.loc[remaining_candidates].sort_values(ascending=True)
            extra = list(remaining_strength.index[:remaining_slots])
            selected.update(extra)

        reduced = list(selected)

        self.reduced_universe = reduced
        # azzera l'eventuale selector_universe perché dipende da K
        self.selector_universe = []

        return reduced

    # ---------- STEP 1 (nuovo): PRE-FILTRO QUANTITATIVO PER IL SELETTORE ----------

    def _build_selector_universe(
        self,
        K: int,
        base_universe: List[str],
    ) -> List[str]:
        """
        Step 1 – Pre-filtro “quantitativo” sui candidati per il selettore.

        Per ogni ticker in base_universe calcola:
            - mu_i (da self.current_mu)
            - sigma_i (sqrt(diagonale di self.current_Sigma_sh)
            - rating_score_norm (rating normalizzato in [0,1] sui soli rated)

        Definisce un asset_score_i semplice:

            asset_score_i = a1 * mu_i - a2 * sigma_i + a3 * rating_score_norm

        Ordina i titoli per asset_score decrescente e taglia a:
            selector_universe = primi min(max(4*K, 40), len(base_universe))

        Il risultato viene salvato in self.selector_universe e ritornato.
        """
        if not base_universe:
            self.selector_universe = []
            return []

        if self.current_mu is None or self.current_Sigma_sh is None:
            # fallback: nessuna informazione → usa l'universo così com'è
            self.selector_universe = list(base_universe)
            return self.selector_universe

        mu_series: pd.Series = self.current_mu
        Sigma_df: pd.DataFrame = self.current_Sigma_sh

        asset_data: List[tuple[str, float, float, float | None]] = []
        rating_values: List[float] = []

        # Prima passata: raccogli mu, sigma, rating
        for t in base_universe:
            # mu_i
            if t in mu_series.index and pd.notna(mu_series.loc[t]):
                mu_i = float(mu_series.loc[t])
            else:
                mu_i = 0.0

            # sigma_i dalla diagonale della covarianza shrinkata
            if t in Sigma_df.index and t in Sigma_df.columns and pd.notna(Sigma_df.loc[t, t]):
                var_i = float(Sigma_df.loc[t, t])
                sigma_i = np.sqrt(var_i) if var_i > 0 else 0.0
            else:
                sigma_i = 0.0

            stock = self.stocks.get(t)
            r = stock.rating_score if stock is not None else None
            if r is not None:
                rating_values.append(r)

            asset_data.append((t, mu_i, sigma_i, r))

        # Normalizzazione rating in [0,1] sui soli titoli con rating
        if rating_values:
            r_min = min(rating_values)
            r_max = max(rating_values)
            denom_r = (r_max - r_min) if (r_max > r_min) else 1.0
        else:
            r_min = 0.0
            denom_r = 1.0

        # Pesi per la combinazione (semplici costanti, modificabili in futuro)
        a1 = 1.0   # peso su mu
        a2 = 0.5   # penalità su sigma
        a3 = 0.3   # bonus su rating normalizzato

        asset_scores: Dict[str, float] = {}
        for t, mu_i, sigma_i, r in asset_data:
            if r is not None:
                rating_norm = (r - r_min) / denom_r
            else:
                rating_norm = 0.0

            score = a1 * mu_i - a2 * sigma_i + a3 * rating_norm
            asset_scores[t] = float(score)

        # Ordina per asset_score decrescente
        sorted_tickers = sorted(
            base_universe,
            key=lambda x: asset_scores.get(x, float("-inf")),
            reverse=True,
        )

        # Dimensione massima per la combinatoria:
        # - almeno 40 titoli
        # - oppure 4*K se più grande di 40
        if K <= 0:
            max_dim = len(sorted_tickers)
        else:
            max_dim = max(4 * K, 40)

        final_dim = min(max_dim, len(sorted_tickers))
        selector_universe = sorted_tickers[:final_dim]

        self.selector_universe = selector_universe
        return selector_universe

    # ---------- STEP 5: OTTIMIZZAZIONE PORTAFOGLIO ----------

    def optimize_portfolio(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_reduced_universe: bool = True,
    ) -> tuple[list[str] | None, Dict[str, float], float | None]:
        """
        Esegue la selezione combinatoria del portafoglio su:
            - selector_universe (pre-filtro quantitativo su U' o sull'universo pieno),
            - se use_reduced_universe=True e self.reduced_universe non è vuoto, usa U' come base,
            - altrimenti usa current_rho.columns come base.

        Parametri:
            - params: dict con K, rating_min, max_unrated_share, alpha, beta, gamma, delta, ecc.
                      Se None, usa PortfolioSelector.build_default_params().
            - use_reduced_universe: se True prova a usare self.reduced_universe come base.
        """

        if self.current_rho is None or self.current_mu is None:
            raise RuntimeError(
                "current_rho / current_mu non disponibili. "
                "Chiama prima estimate_risk_window(), poi (eventualmente) build_reduced_universe()."
            )

        rho = self.current_rho
        mu_series = self.current_mu

        # Parametri di default se non forniti
        if params is None:
            params = PortfolioSelector.build_default_params()

        # Scegli l'universo base: U' se disponibile, altrimenti tutto
        if use_reduced_universe and self.reduced_universe:
            base_universe = [t for t in self.reduced_universe if t in rho.columns and t in mu_series.index]
        else:
            base_universe = [t for t in rho.columns if t in mu_series.index]

        if not base_universe:
            raise ValueError("Nessun titolo disponibile per l'ottimizzazione (universo vuoto).")

        # Step 1: pre-filtro quantitativo per il selettore
        K = int(params.get("K", 0))
        universe = self._build_selector_universe(K, base_universe)

        if not universe:
            raise ValueError("selector_universe vuoto dopo il pre-filtro quantitativo.")

        # Dizionari rating_scores, sectors, has_rating, mu per l'universo scelto
        rating_scores: Dict[str, float | None] = {}
        sectors: Dict[str, str | None] = {}
        has_rating: Dict[str, bool] = {}
        mu_dict: Dict[str, float] = {}

        for t in universe:
            stock = self.stocks.get(t)
            if stock is not None:
                rating_scores[t] = stock.rating_score
                sectors[t] = stock.sector
                has_rating[t] = (stock.rating_score is not None)
            else:
                rating_scores[t] = None
                sectors[t] = None
                has_rating[t] = False

            # mu corrente per il titolo (se presente)
            if t in mu_series.index and pd.notna(mu_series.loc[t]):
                mu_dict[t] = float(mu_series.loc[t])
            else:
                mu_dict[t] = 0.0

        # Rho ridotta all'universo scelto
        rho_sub = rho.loc[universe, universe]

        # Euristica di ordinamento candidati:
        #   - prima i titoli con rating (has_rating=True),
        #   - poi in ordine decrescente di rating_score,
        #   - a parità, in ordine decrescente di mu atteso.
        def sort_key(t: str):
            hr = has_rating.get(t, False)
            rs = rating_scores.get(t)
            rs_val = rs if rs is not None else -1e9
            mu_val = mu_dict.get(t, 0.0)
            return (0 if hr else 1, -rs_val, -mu_val)

        candidati = sorted(universe, key=sort_key)

        # Esecuzione del selettore combinatorio (OOP)
        selector = PortfolioSelector(
            rho=rho_sub,
            rating_scores=rating_scores,
            sectors=sectors,
            has_rating=has_rating,
            mu=mu_dict,
            params=params,
        )
        best_subset, best_score = selector.select(candidati=candidati)

        if best_subset is None or len(best_subset) == 0:
            # Nessuna soluzione che rispetti i vincoli
            self.last_portfolio_tickers = []
            self.last_portfolio_weights = {}
            self.last_portfolio_score = None
            return None, {}, None

        # ============================
        # STEP 6: calcolo pesi
        # ============================
        weights = self._compute_weights_for_subset(list(best_subset), params)

        self.last_portfolio_tickers = list(best_subset)
        self.last_portfolio_weights = weights
        self.last_portfolio_score = float(best_score)

        return best_subset, weights, float(best_score)

    # ============================
    # METODI PESI PORTAFOGLIO (STEP 6)
    # ============================

    def _round_weights_to_2_decimals(
        self,
        tickers: list[str],
        w: np.ndarray,
    ) -> dict[str, float]:
        """
        Arrotonda i pesi alla seconda cifra decimale cercando di mantenere
        la somma a 1.0 (entro gli inevitabili limiti di arrotondamento).
        """
        w = np.asarray(w, dtype=float)

        # Prima normalizziamo a somma 1 (per sicurezza)
        s = w.sum()
        if s <= 0:
            w = np.ones(len(tickers)) / len(tickers)
        else:
            w = w / s

        # Arrotonda a 2 decimali
        w_rounded = np.round(w, 2)

        # Aggiusta eventuale differenza sulla somma
        diff = 1.0 - w_rounded.sum()
        if abs(diff) > 1e-6:
            j = int(np.argmax(w_rounded))
            w_rounded[j] += diff

        return {t: float(w_i) for t, w_i in zip(tickers, w_rounded)}

    def _equal_weights(self, tickers: list[str]) -> dict[str, float]:
        """
        Assegna pesi uguali a tutti i ticker del portafoglio.

        Usato come fallback quando la mean-variance è impossibile o instabile.
        """
        n = len(tickers)
        if n == 0:
            return {}
        w = 1.0 / n
        return {t: w for t in tickers}

    def _mean_variance_weights(
        self,
        tickers: list[str],
        risk_aversion: float = 1.0,
        allow_short: bool = False,
        ridge: float = 1e-4,
    ) -> dict[str, float]:
        """
        Calcola pesi mean-variance (Markowitz) per i titoli in 'tickers',
        usando self.current_mu e self.current_Sigma_sh.
        """
        n = len(tickers)
        if n == 0:
            return {}

        if self.current_mu is None or self.current_Sigma_sh is None:
            return self._equal_weights(tickers)

        tickers = list(tickers)
        mu_series: pd.Series = self.current_mu
        Sigma_df: pd.DataFrame = self.current_Sigma_sh

        missing_mu = [t for t in tickers if t not in mu_series.index]
        missing_Sigma = [t for t in tickers if t not in Sigma_df.index]
        if missing_mu or missing_Sigma:
            return self._equal_weights(tickers)

        mu_vec = mu_series.loc[tickers].astype(float).values
        Sigma_sub = Sigma_df.loc[tickers, tickers].astype(float).values

        Sigma_reg = Sigma_sub + ridge * np.eye(n)
        ones = np.ones(n)

        try:
            A = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            return self._equal_weights(tickers)

        c = risk_aversion * mu_vec
        Ac = A @ c
        A1 = A @ ones
        num = ones @ Ac - 1.0
        den = ones @ A1

        if abs(den) < 1e-12:
            return self._equal_weights(tickers)

        lam = num / den
        w = A @ (c - lam * ones)

        if not allow_short:
            w_min = 1.0 / (2.0 * n)
            w = np.maximum(w, w_min)
            s = w.sum()
            if s <= 0:
                return self._equal_weights(tickers)
            w = w / s

        return self._round_weights_to_2_decimals(tickers, w)

    def _compute_weights_for_subset(
        self,
        best_subset: list[str],
        params: dict,
    ) -> dict[str, float]:
        """
        Wrapper unico per decidere come calcolare i pesi (eq vs mean-variance)
        in base ai parametri.
        """
        if not best_subset:
            return {}

        mode = params.get("weights_mode", "mv")          # "mv" o "eq"
        risk_aversion = float(params.get("mv_risk_aversion", 1.0))
        allow_short = bool(params.get("mv_allow_short", False))

        if mode == "mv":
            return self._mean_variance_weights(
                tickers=best_subset,
                risk_aversion=risk_aversion,
                allow_short=allow_short,
            )
        else:
            return self._equal_weights(best_subset)


if __name__ == "__main__":
    # Test rapido complessivo Step 1–3 + Step 5–6
    import traceback
    import time
    from pprint import pprint

    model = Model()

    try:
        # ============================
        # STEP 1: caricamento dati
        # ============================
        model.load_data_from_dao()

        print("=== TEST MODEL (Step 1: load_data_from_dao) ===")
        print(f"Shape prices_df: {model.prices_df.shape if model.prices_df is not None else None}")
        print(f"Shape ratings_df: {model.ratings_df.shape if model.ratings_df is not None else None}")
        print(f"Num stocks: {len(model.stocks)}")
        print(f"Shape returns_df: {model.returns_df.shape if model.returns_df is not None else None}")
        print(f"Ticker con rating: {len(model.tickers_with_rating)}")
        print(f"Ticker senza rating: {len(model.tickers_without_rating)}")

        if model.prices_df is None or model.ratings_df is None or model.returns_df is None:
            raise AssertionError("Dati non caricati correttamente in Model.")
        if len(model.stocks) == 0:
            raise AssertionError("model.stocks è vuoto.")

        # ============================
        # STEP 2: stima rischio su ~1 anno (prime 252 date)
        # ============================
        returns_df = model.returns_df
        start = returns_df.index[0]
        end = returns_df.index[min(251, len(returns_df) - 1)]

        print("\n=== TEST MODEL (Step 2: estimate_risk_window) ===")
        print(f"Finestra: {start.date()} → {end.date()}")

        model.estimate_risk_window(start, end, shrink_lambda=0.1)

        rho = model.current_rho
        Sigma_sh = model.current_Sigma_sh
        mu = model.current_mu

        print(f"Shape current_rho: {rho.shape if rho is not None else None}")
        print(f"Shape current_Sigma_sh: {Sigma_sh.shape if Sigma_sh is not None else None}")
        print(f"Shape current_mu: {mu.shape if mu is not None else None}")

        if rho is None or Sigma_sh is None or mu is None:
            raise AssertionError("Le matrici di rischio correnti non sono state stimate.")

        # ============================
        # STEP 3: costruzione universo ridotto U'
        # ============================
        print("\n=== TEST MODEL (Step 3: build_reduced_universe) ===")
        reduced = model.build_reduced_universe(
            tau=0.3,
            k=10,
            max_size=60,
            max_unrated_share=0.0
        )

        print(f"Dimensione universo ridotto U': {len(reduced)}")
        print("Primi 10 ticker in U':", reduced[:10])
        print(
            "Numero nodi nel grafo corrente:",
            model.current_graph.number_of_nodes() if model.current_graph is not None else 0
        )
        print(
            "Numero archi nel grafo corrente:",
            model.current_graph.number_of_edges() if model.current_graph is not None else 0
        )

    except AssertionError as e:
        print("\nTEST MODEL STEP 1–3 FALLITO:")
        print(" -", e)

    except Exception as e:
        print("\nERRORE IMPREVISTO DURANTE IL TEST MODEL (Step 1–3):")
        traceback.print_exc()

    else:
        print("\nTEST MODEL STEP 1–3 OK.")

        # ============================
        # STEP 5–6: optimize_portfolio + pesi
        # ============================
        print("\n=== TEST MODEL (Step 5–6: optimize_portfolio + weights) ===")

        if model.reduced_universe:
            print(f"Ticker in U' (reduced_universe): {len(model.reduced_universe)}")
        else:
            print("Attenzione: reduced_universe è vuoto, userò l'universo completo (current_rho.columns).")

        params = PortfolioSelector.build_default_params()
        params["K"] = 4
        params["max_unrated_share"] = 0.2
        params["rating_min"] = 13.0
        params["max_share_per_sector"] = 0.5

        params["weights_mode"] = "mv"
        params["mv_risk_aversion"] = 1.0
        params["mv_allow_short"] = False

        print("Parametri ottimizzazione:")
        print(f"  K = {params['K']}")
        print(f"  max_unrated_share = {params['max_unrated_share']}")
        print(f"  rating_min = {params['rating_min']}")
        print(f"  max_share_per_sector = {params['max_share_per_sector']}")
        print(f"  weights_mode = {params['weights_mode']}")
        print(f"  mv_risk_aversion = {params['mv_risk_aversion']}")
        print(f"  mv_allow_short = {params['mv_allow_short']}")
        print("Avvio optimize_portfolio(...)\n", flush=True)

        t0 = time.time()
        try:
            best_subset, weights, best_score = model.optimize_portfolio(
                params=params,
                use_reduced_universe=True
            )
        except Exception:
            t1 = time.time()
            print(f"ERRORE durante optimize_portfolio dopo {t1 - t0:.2f} secondi:")
            traceback.print_exc()
        else:
            t1 = time.time()
            print(f"optimize_portfolio terminato in {t1 - t0:.2f} secondi.\n")

            print(f"Dim selector_universe: {len(model.selector_universe)}")

            if best_subset is None:
                print("Nessuna soluzione trovata con i vincoli correnti.")
            else:
                print(f"Portafoglio ottimo (len={len(best_subset)}): {best_subset}")
                print(f"Score combinatorio: {best_score}")
                print("Pesi assegnati:")
                pprint(weights)
                print("Somma pesi:", sum(weights.values()))
