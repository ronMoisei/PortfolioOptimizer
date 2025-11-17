# controller.py
from __future__ import annotations
import flet as ft
import numpy as np
import pandas as pd
import networkx as nx

# Importa le tue classi Model e PortfolioSelector
from model.model import Model
from model.selector import PortfolioSelector


class Controller:
    def __init__(self, view: ft.Page, model: Model):
        self._view = view
        self._model = model

    # ------------------------------------------------------------------ #
    # HANDLER PRINCIPALE: OTTIMIZZA PORTAFOGLIO
    # ------------------------------------------------------------------ #
    def handle_optimize(self, e):
        """
        Handler principale che orchestra l'intera pipeline
        in base agli input dell'utente (come da tesi).
        """
        try:
            self._view.clear_tabs()

            # 1. Validazione e Lettura Input GUI
            try:
                capital = float(self._view._txtCapital.value or "100000")
                if capital <= 0:
                    raise ValueError("Il capitale deve essere positivo.")
            except ValueError:
                raise ValueError("Capitale iniziale non valido (es: 100000).")

            risk_profile = self._view._ddRiskProfile.value
            if not risk_profile:
                raise ValueError("Seleziona un profilo di rischio.")

            horizon = int(self._view._ddHorizon.value or "10")

            # 2. Controllo stato modello
            if self._model.current_rho is None or self._model.current_mu is None:
                raise RuntimeError(
                    "Dati non pronti. Esegui prima il caricamento e la stima "
                    "del modello (Passi 1-3) nel backend."
                )

            # 3. Traduci il Rischio in Parametri (Logica Tesi)
            params = self._translate_risk_profile(risk_profile)
            params["weights_mode"] = "mv"  # Usa sempre Mean-Variance

            # 4. Esegui Ottimizzazione (Model)
            use_reduced = bool(self._model.reduced_universe)
            best_subset, weights, best_score = self._model.optimize_portfolio(
                params=params,
                use_reduced_universe=use_reduced,
            )

            if not best_subset:
                raise RuntimeError("Nessuna soluzione trovata per i vincoli correnti.")

            # 5. Calcola Metriche e Proiezioni (Controller)
            metrics = self._compute_projections_and_metrics(
                best_subset, weights, capital, horizon
            )

            # 6. Calcola Metriche Grafo (Controller)
            graph_metrics = self._compute_graph_metrics()

            # 7. Mostra Risultati nei Tab
            self._show_dashboard_results(metrics, params)
            self._show_composition(best_subset, weights)
            self._show_graph_analysis(graph_metrics)

            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore: {ex}")

    # ------------------------------------------------------------------ #
    # LOGICA "TESI": TRADURRE IL PROFILO DI RISCHIO IN PARAMETRI
    # ------------------------------------------------------------------ #
    def _translate_risk_profile(self, profile: str) -> dict:
        """
        Questa è la logica di business centrale della tesi.
        Traduce un input utente ("Basso", "Medio", "Alto")
        in parametri tecnici per il PortfolioSelector.
        """
        params = PortfolioSelector.build_default_params()

        if profile == "Basso":
            # Obiettivo: Massima diversificazione, alta qualità, basso rischio
            params["K"] = 30  # K alto per diversificare
            params["rating_min"] = 15.0  # Almeno BBB+
            params["max_unrated_share"] = 0.0  # Solo titoli con rating
            params["alpha"] = 2.0  # Peso alto su bassa correlazione
            params["beta"] = 1.5  # Peso alto su rating
            params["gamma"] = 1.0  # Peso alto su penalità settore
            params["delta"] = 0.1  # Peso basso su rendimento atteso
            params["mv_risk_aversion"] = 0.5  # Bassa avversione (punta a min-vol)

        elif profile == "Medio":
            # Obiettivo: Bilanciato
            params["K"] = 20  # K medio
            params["rating_min"] = 13.0  # Almeno BBB-
            params["max_unrated_share"] = 0.1  # Max 10% unrated
            params["alpha"] = 1.0
            params["beta"] = 1.0
            params["gamma"] = 0.5
            params["delta"] = 1.0  # Pesi bilanciati
            params["mv_risk_aversion"] = 1.0  # Standard

        elif profile == "Alto":
            # Obiettivo: Massimizzare rendimento atteso, alta concentrazione
            params["K"] = 10  # K basso (concentrato)
            params["rating_min"] = 0.0  # Qualsiasi rating
            params["max_unrated_share"] = 0.5  # Fino a 50% unrated
            params["alpha"] = 0.1  # Bassa importanza a correlazione
            params["beta"] = 0.1  # Bassa importanza a rating
            params["gamma"] = 0.0
            params["delta"] = 2.0  # Peso massimo su rendimento atteso
            params["mv_risk_aversion"] = 2.0  # Alta avversione (cerca più rendimento)

        return params

    # ------------------------------------------------------------------ #
    # LOGICA "TESI": CALCOLO METRICHE E PROIEZIONI
    # ------------------------------------------------------------------ #
    def _compute_projections_and_metrics(
            self, tickers: list[str], weights_dict: dict, capital: float, horizon: int
    ) -> dict:
        """
        Calcola metriche finanziarie chiave (Sharpe, Vol, CAGR)
        e proiezioni di valore futuro.
        """
        if (self._model.current_mu is None) or (self._model.current_Sigma_sh is None):
            raise RuntimeError("Dati di rischio (mu, Sigma) non disponibili nel modello.")

        mu_series = self._model.current_mu
        sigma_df = self._model.current_Sigma_sh

        # Prepara vettori
        weights = np.array([weights_dict.get(t, 0.0) for t in tickers])

        # Sotto-matrici di rischio allineate
        mu_vec = mu_series.loc[tickers].values
        sigma_sub = sigma_df.loc[tickers, tickers].values

        # 1. Calcola Mu e Volatilità di Portafoglio (Daily)
        # Nota: mu_vec sono rendimenti log, quindi la somma è corretta
        mu_port_daily_log = float(weights @ mu_vec)

        # Volatilità (sqrt(w' * Sigma * w))
        var_port_daily = float(weights @ sigma_sub @ weights.T)
        sigma_port_daily = np.sqrt(var_port_daily) if var_port_daily > 0 else 0.0

        # 2. Annualizza (252 giorni di trading)
        # CAGR (Rendimento atteso annualizzato)
        cagr = np.exp(mu_port_daily_log * 252) - 1

        # Volatilità annualizzata
        sigma_annual = sigma_port_daily * np.sqrt(252)

        # 3. Sharpe Ratio (assumendo Risk-Free Rate = 0)
        sharpe = (cagr / sigma_annual) if sigma_annual > 1e-9 else 0.0

        # 4. Proiezioni (Log-Normal Distribution)
        # Valore atteso (FV)
        fv_atteso = capital * ((1 + cagr) ** horizon)

        # Intervalli di confidenza (95%)
        T = horizon
        mu_log = mu_port_daily_log * 252
        sigma_log = sigma_annual  # Volatilità del rendimento log annualizzato

        # Calcolo quantile 95% (z=1.96) su distribuzione log-normale
        log_fv_mean = (mu_log - 0.5 * sigma_log ** 2) * T + np.log(capital)
        log_fv_std = sigma_log * np.sqrt(T)

        fv_low_95 = np.exp(log_fv_mean - 1.96 * log_fv_std)
        fv_high_95 = np.exp(log_fv_mean + 1.96 * log_fv_std)

        return {
            "capital": capital,
            "horizon": horizon,
            "cagr": cagr,
            "sigma_annual": sigma_annual,
            "sharpe": sharpe,
            "fv_atteso": fv_atteso,
            "fv_low_95": fv_low_95,
            "fv_high_95": fv_high_95,
        }

    # ------------------------------------------------------------------ #
    # LOGICA "TESI": ANALISI GRAFO
    # ------------------------------------------------------------------ #
    def _compute_graph_metrics(self) -> dict:
        """
        Estrae metriche di base dal grafo (come da tesi).
        """
        graph = self._model.current_graph
        if graph is None:
            return {"error": "Grafo non calcolato."}

        try:
            # Calcola cluster (componenti connesse)
            clusters = list(nx.connected_components(graph))
            clusters_sorted = sorted(clusters, key=len, reverse=True)

            # Calcola centralità (Degree Centrality)
            centrality = nx.degree_centrality(graph)

            # Top 5 nodi centrali
            top_5_central = sorted(centrality.items(), key=lambda item: item[1], reverse=True)[:5]

            return {
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "n_clusters": len(clusters_sorted),
                "top_5_clusters": [list(c) for c in clusters_sorted[:5]],
                "top_5_central_nodes": top_5_central,
            }
        except Exception as e:
            return {"error": f"Errore analisi grafo: {e}"}

    # ------------------------------------------------------------------ #
    # VISUALIZZAZIONE RISULTATI NEI TABS
    # ------------------------------------------------------------------ #

    def _show_dashboard_results(self, metrics: dict, params: dict):
        """ Popola il Tab 'Dashboard' con le proiezioni. """

        def f_num(n):
            """ Formatta un numero in euro """
            return f"{n:,.2f} €"

        self._view.tab_dashboard.controls.append(
            ft.Text("Dashboard Risultati", size=18, weight="bold")
        )
        self._view.tab_dashboard.controls.append(ft.Divider())
        self._view.tab_dashboard.controls.append(
            ft.Text(f"Proiezione a {metrics['horizon']} anni (Capitale: {f_num(metrics['capital'])})")
        )

        self._view.tab_dashboard.controls.append(
            ft.Text(
                f"Valore Atteso: {f_num(metrics['fv_atteso'])}",
                size=16, weight="bold", color="green"
            )
        )
        self._view.tab_dashboard.controls.append(
            ft.Text(f"CAGR (Rendimento annuo atteso): {metrics['cagr']:.2%}")
        )
        self._view.tab_dashboard.controls.append(
            ft.Text(f"Volatilità annua: {metrics['sigma_annual']:.2%}")
        )
        self._view.tab_dashboard.controls.append(
            ft.Text(f"Sharpe Ratio (R/V): {metrics['sharpe']:.3f}")
        )

        self._view.tab_dashboard.controls.append(ft.Divider())
        self._view.tab_dashboard.controls.append(
            ft.Text(f"Intervallo di confidenza (95%):")
        )
        self._view.tab_dashboard.controls.append(
            ft.Text(f"  Basso: {f_num(metrics['fv_low_95'])}")
        )
        self._view.tab_dashboard.controls.append(
            ft.Text(f"  Alto: {f_num(metrics['fv_high_95'])}")
        )

        self._view.tab_dashboard.controls.append(ft.Divider())
        self._view.tab_dashboard.controls.append(
            ft.Text("Parametri usati per questo profilo:")
        )
        self._view.tab_dashboard.controls.append(
            ft.Text(f"  K={params['K']}, Rating Min={params['rating_min']}, "
                    f"Max Unrated={params['max_unrated_share']:.0%}")
        )

    def _show_composition(self, tickers: list[str], weights: dict):
        """ Popola il Tab 'Composizione'. """

        self._view.tab_composition.controls.append(
            ft.Text(f"Composizione Portafoglio ({len(tickers)} Titoli)", size=18, weight="bold")
        )
        self._view.tab_composition.controls.append(ft.Divider())

        stocks = self._model.stocks
        mu_series = self._model.current_mu

        # Ordina per peso
        sorted_tickers = sorted(tickers, key=lambda t: weights.get(t, 0.0), reverse=True)

        for t in sorted_tickers:
            stock = stocks.get(t)
            sector = stock.sector if stock and stock.sector else "N/A"
            rs = stock.rating_score if stock else None
            rating_text = "unrated" if rs is None else f"{rs:.1f}"

            w_i = float(weights.get(t, 0.0))

            if mu_series is not None and t in mu_series.index:
                mu_i_daily = float(mu_series.loc[t])
                cagr_i = np.exp(mu_i_daily * 252) - 1
            else:
                cagr_i = 0.0

            line = (
                f"[{w_i:6.2%}] {t:5} | "
                f"Settore: {sector:15} | "
                f"Rating: {rating_text:7} | "
                f"CAGR atteso: {cagr_i:.2%}"
            )
            self._view.tab_composition.controls.append(ft.Text(line, font_family="monospace"))

    def _show_graph_analysis(self, metrics: dict):
        """ Popola il Tab 'Analisi Grafo'. """

        self._view.tab_graph.controls.append(
            ft.Text("Analisi Rete di Correlazione", size=18, weight="bold")
        )
        self._view.tab_graph.controls.append(ft.Divider())

        if "error" in metrics:
            self._view.tab_graph.controls.append(ft.Text(metrics['error'], color="red"))
            return

        self._view.tab_graph.controls.append(
            ft.Text(f"Nodi: {metrics['n_nodes']}, Archi: {metrics['n_edges']}")
        )
        self._view.tab_graph.controls.append(
            ft.Text(f"Numero di Cluster (Componenti Connesse): {metrics['n_clusters']}")
        )

        self._view.tab_graph.controls.append(ft.Divider())
        self._view.tab_graph.controls.append(
            ft.Text("Top 5 Nodi Centrali (Degree Centrality):")
        )
        for node, centrality in metrics['top_5_central_nodes']:
            self._view.tab_graph.controls.append(
                ft.Text(f"  - {node} (Score: {centrality:.3f})")
            )

        self._view.tab_graph.controls.append(ft.Divider())
        self._view.tab_graph.controls.append(
            ft.Text("Top 5 Cluster (per dimensione):")
        )
        for i, cluster in enumerate(metrics['top_5_clusters']):
            self._view.tab_graph.controls.append(
                ft.Text(f"  Cluster {i + 1} ({len(cluster)} nodi): {', '.join(cluster[:5])}...")
            )

    # ------------------------------------------------------------------ #
    # GESTIONE ERRORI
    # ------------------------------------------------------------------ #
    def _show_error(self, message: str):
        """Mostra un messaggio di errore in tutti i tabs."""
        self._view.clear_tabs()
        self._view.tab_dashboard.controls.append(ft.Text(message, color="red"))
        self._view.update_page()