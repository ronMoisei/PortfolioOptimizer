from __future__ import annotations

from typing import Dict, Optional, List, Any

import pandas as pd
import networkx as nx

from database.DAO import DAO
from model.stock import Stock
from model.risk_estimator import RiskEstimator
from model.graph_builder import GraphBuilder
from model.selector import PortfolioSelector
from model.portfolio_weights import PortfolioWeights


class Model:
    """
    Classe principale del modello.

    Responsabilità:
        - carica dati con il DAO
        - costruisce prices_df, ratings_df, stocks, returns_df
        - stima rho, Sigma_sh, mu sull'intera storia disponibile
        - costruisce il grafo di correlazione (GraphBuilder)
        - applica filtri (soglia, k-NN) e costruisce un universo ridotto U'
        - delega la selezione combinatoria del portafoglio a PortfolioSelector
        - calcola i pesi (PortfolioWeights)
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
        self.map_has_rating: Dict[str, bool] = {}

        # Grafo e universo ridotto
        self.current_graph: nx.Graph | None = None
        self.reduced_universe: List[str] = []

        # Universo usato dal selettore (dopo pre-filtro quantitativo)
        self.selector_universe: List[str] = []

        # Risultati ultima ottimizzazione
        self.last_portfolio_tickers: list[str] = []
        self.last_portfolio_weights: Dict[str, float] = {}
        self.last_portfolio_score: float | None = None

    # ---------- CARICAMENTO DATI ----------

    def load_data_from_dao(self) -> None:
        """
        Carica i dati di base (prezzi, rating, oggetti Stock) dal DAO
        e costruisce returns_df, più alcune liste di supporto sui rating.
        """
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

        # mappatura per controlli rapidi
        self.map_has_rating = {
            t: (s.rating_score is not None)
            for t, s in self.stocks.items()
        }

    # ---------- STIMA RISCHIO SULL'INTERA STORIA ----------

    def estimate_risk(
        self,
        shrink_lambda: float = 0.1,
        min_non_na_ratio: float = 0.8,
        winsor_lower: float | None = 0.01,
        winsor_upper: float | None = 0.99,
    ) -> None:
        """
        Stima rho, Sigma_sh, mu.

        Usa RiskEstimator.estimate_from_returns come "libreria".
        """
        if self.returns_df is None:
            raise RuntimeError(
                "returns_df non è stato caricato. Chiama load_data_from_dao() prima."
            )

        rho, Sigma_sh, mu = RiskEstimator.estimate_from_returns(
            self.returns_df,
            shrink_lambda=shrink_lambda,
            min_non_na_ratio=min_non_na_ratio,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
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

    # ---------- GRAFO E UNIVERSO RIDOTTO ----------

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

        Passi:
        1) parte da current_rho,
        2) restringe eventualmente ai soli titoli con rating se max_unrated_share == 0,
        3) usa GraphBuilder.build_filtered_graph per costruire grafo, adj e dist_kNN,
        4) calcola la strength di ogni nodo,
        5) seleziona U' bilanciando pool rated / unrated e privilegiando strength bassa.
        """
        if self.current_rho is None:
            raise RuntimeError(
                "current_rho non disponibile. Chiama estimate_risk() prima di build_reduced_universe()."
            )

        rho_full = self.current_rho.copy()
        tickers_all = list(rho_full.columns)

        # --- 1) Restrizione iniziale in base a max_unrated_share ---

        if max_unrated_share == 0.0:
            # Universo iniziale: solo titoli con rating (indipendentemente dal valore)
            base_tickers = [
                t for t in tickers_all if self.map_has_rating.get(t, False)
            ]
            if not base_tickers:
                raise ValueError(
                    "Nessun titolo con rating disponibile nell'universo corrente."
                )
            rho = rho_full.loc[base_tickers, base_tickers]
        else:
            base_tickers = tickers_all
            rho = rho_full.loc[base_tickers, base_tickers]

        # --- 2) Costruzione grafo + adj + dist_knn tramite GraphBuilder ---

        G, adj, dist_knn = GraphBuilder.build_filtered_graph(
            rho=rho,
            tau=tau,
            k=k,
            signed=False,
        )
        self.current_graph = G

        # --- 3) Calcolo strength e definizione dei due pool ---

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

        # --- 4) Quote per U': N_Rated, N_Unrated ---

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

        # --- 5) Selezione effettiva per strength bassa ---

        rated_strength = (
            strength.loc[pool_rated].sort_values(ascending=True)
            if pool_rated
            else pd.Series(dtype=float)
        )
        unrated_strength = (
            strength.loc[pool_unrated].sort_values(ascending=True)
            if pool_unrated
            else pd.Series(dtype=float)
        )

        selected_rated = list(rated_strength.index[:N_rated])
        selected_unrated = list(unrated_strength.index[:N_unrated])

        selected = set(selected_rated + selected_unrated)

        # se mancano ancora titoli per arrivare a max_size,
        # riempio con i restanti (sempre per strength bassa)
        total_selected = len(selected)
        if total_selected < max_size:
            remaining_slots = max_size - total_selected

            remaining_candidates = [
                t for t in all_candidates if t not in selected
            ]
            if remaining_candidates:
                remaining_strength = strength.loc[
                    remaining_candidates
                ].sort_values(ascending=True)
                extra = list(remaining_strength.index[:remaining_slots])
                selected.update(extra)

        reduced = list(selected)

        self.reduced_universe = reduced
        # azzera l'eventuale selector_universe perché dipende da K
        self.selector_universe = []

        return reduced

    # ---------- OTTIMIZZAZIONE PORTAFOGLIO ----------

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
                "Chiama prima estimate_risk(), poi (eventualmente) build_reduced_universe()."
            )

        rho = self.current_rho
        mu_series = self.current_mu

        # Parametri di default se non forniti
        if params is None:
            params = PortfolioSelector.build_default_params()

        # Scegli l'universo base: U' se disponibile, altrimenti tutto
        if use_reduced_universe and self.reduced_universe:
            base_universe = [
                t
                for t in self.reduced_universe
                if t in rho.columns and t in mu_series.index
            ]
        else:
            base_universe = [
                t for t in rho.columns if t in mu_series.index
            ]

        if not base_universe:
            raise ValueError(
                "Nessun titolo disponibile per l'ottimizzazione (universo vuoto)."
            )

        # ---------- STEP 1: pre-filtro quantitativo (nel Selector) ----------

        K = int(params.get("K", 0))
        universe = PortfolioSelector.build_selector_universe(
            base_universe=base_universe,
            K=K,
            mu=self.current_mu,
            Sigma_sh=self.current_Sigma_sh,
            stocks=self.stocks,
        )
        self.selector_universe = list(universe)

        if not universe:
            raise ValueError(
                "selector_universe vuoto dopo il pre-filtro quantitativo."
            )

        # ---------- STEP 2: costruzione dizionari per il Selector ----------

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

        # ---------- STEP 3: ordinamento candidati (nel Selector) ----------

        candidati = PortfolioSelector.sort_candidates(
            tickers=universe,
            rating_scores=rating_scores,
            has_rating=has_rating,
            mu=mu_dict,
        )

        # ---------- STEP 4: selezione combinatoria ----------

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

        # ---------- STEP 5: calcolo pesi (solo long) ----------

        mode = params.get("weights_mode", "mv")          # "mv" o "eq"
        risk_aversion = float(params.get("mv_risk_aversion", 1.0))

        weights = PortfolioWeights.compute(
            tickers=list(best_subset),
            mode=mode,
            mu=self.current_mu,
            Sigma_sh=self.current_Sigma_sh,
            risk_aversion=risk_aversion,
        )

        self.last_portfolio_tickers = list(best_subset)
        self.last_portfolio_weights = weights
        self.last_portfolio_score = float(best_score)

        return list(best_subset), weights, float(best_score)


if __name__ == "__main__":
    # Test rapido complessivo
    import traceback
    import time

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
        # STEP 2: stima rischio su tutta la storia
        # ============================
        print("\n=== TEST MODEL (Step 2: estimate_risk) ===")
        model.estimate_risk(shrink_lambda=0.1)

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
        params["K"] = 5
        params["max_unrated_share"] = 0.2
        params["rating_min"] = 13.0
        params["max_share_per_sector"] = 0.5

        params["weights_mode"] = "mv"
        params["mv_risk_aversion"] = 1.0

        print("Parametri ottimizzazione:")
        print(f"  K = {params['K']}")
        print(f"  max_unrated_share = {params['max_unrated_share']}")
        print(f"  rating_min = {params['rating_min']}")
        print(f"  max_share_per_sector = {params['max_share_per_sector']}")
        print(f"  weights_mode = {params['weights_mode']}")
        print(f"  mv_risk_aversion = {params['mv_risk_aversion']}")
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
                print(weights)
                print("Somma pesi:", sum(weights.values()))
