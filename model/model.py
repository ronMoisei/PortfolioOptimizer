import csv
import math
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx


class Model:
    """Dominio applicativo per la costruzione del grafo finanziario."""

    _RATING_SCALE_ORDER = [
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "CCC+",
        "CCC",
        "CCC-",
        "CC",
        "C",
        "D",
    ]

    def __init__(self):
        # dataset
        self._price_series: Dict[str, Dict[date, float]] = defaultdict(dict)
        self._sorted_dates: List[date] = []
        self._ratings: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
        self._agency_lookup: Dict[str, str] = {}
        self._store_cache: Optional[List[Tuple[str, str]]] = None

        # modelli derivati
        self._graph: Optional[nx.Graph] = None
        self._current_agency: Optional[str] = None
        self._last_window: Optional[Tuple[date, date]] = None
        self._selected_dates: List[date] = []

        # parametri
        self._max_path_length = 6
        self._min_weight_threshold = 0.05
        self._rating_weight = 0.6

        self._rating_score = {
            label: float(len(self._RATING_SCALE_ORDER) - idx)
            for idx, label in enumerate(self._RATING_SCALE_ORDER)
        }
        self._max_rating_score = max(self._rating_score.values())

        # caricamento dati
        self._load_price_series()
        self._load_ratings()

    # ------------------------------------------------------------------
    # caricamento dati
    def _load_price_series(self) -> None:
        dates = set()
        with open("database/all_stocks_5yr.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row["Name"].strip().upper()
                try:
                    date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                    close = float(row["close"])
                except (ValueError, TypeError):
                    continue
                if close <= 0:
                    continue
                self._price_series[ticker][date] = close
                dates.add(date)
        self._sorted_dates = sorted(dates)

    def _load_ratings(self) -> None:
        with open(
            "database/corporateCreditRatingWithFinancialRatios.csv",
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)
            for row in reader:
                agency = row["Rating Agency"].strip()
                ticker = row["Ticker"].strip().upper()
                rating_label = row["Rating"].strip().upper().replace(" ", "")
                if rating_label not in self._rating_score:
                    continue
                try:
                    rating_date = datetime.strptime(row["Rating Date"], "%Y-%m-%d").date()
                except (TypeError, ValueError):
                    continue
                sector = row.get("Sector", "").strip() or "Unknown"
                score = self._rating_score[rating_label]

                agency_data = self._ratings[agency]
                current = agency_data.get(ticker)
                if current is None or rating_date >= current["date"]:
                    agency_data[ticker] = {
                        "rating": rating_label,
                        "score": score,
                        "date": rating_date,
                        "sector": sector,
                    }

    # ------------------------------------------------------------------
    # API pubblico
    def getStores(self) -> List[Tuple[str, str]]:
        """Restituisce le agenzie di rating disponibili."""
        if self._store_cache is not None:
            return list(self._store_cache)

        options: List[Tuple[str, str]] = []
        self._agency_lookup.clear()

        agencies = sorted(self._ratings.keys())
        idx = 0
        for agency in agencies:
            tickers = {
                ticker
                for ticker in self._ratings[agency]
                if ticker in self._price_series
            }
            if len(tickers) < 2:
                continue
            label = f"{agency} ({len(tickers)} titoli)"
            code = str(idx)
            options.append((code, label))
            self._agency_lookup[code] = agency
            idx += 1

        if not options:
            raise ValueError("Nessuna agenzia di rating disponibile con dati di prezzo")

        self._store_cache = options
        return list(options)

    def buildGraph(self, store_value: str, k_days: int) -> None:
        if not store_value:
            raise ValueError("Valore di store non valido")
        if k_days < 2:
            raise ValueError("K deve essere almeno 2 per calcolare le correlazioni")
        if k_days > len(self._sorted_dates):
            raise ValueError("K troppo grande rispetto all'orizzonte temporale disponibile")

        store_code = store_value.split("-", 1)[0]
        if store_code not in self._agency_lookup:
            raise ValueError("Store selezionato non valido")
        agency = self._agency_lookup[store_code]
        returns = self._build_return_matrix(agency, k_days)
        tickers = list(returns.keys())
        if len(tickers) < 2:
            raise ValueError(
                "Numero insufficiente di titoli con dati completi per costruire il grafo"
            )

        correlations = self._compute_correlations(returns)
        graph = nx.Graph()

        # aggiungi nodi con attributi
        for ticker in tickers:
            rating_info = self._ratings[agency][ticker]
            graph.add_node(
                ticker,
                rating_score=rating_info["score"],
                rating=rating_info["rating"],
                rating_date=rating_info["date"],
                sector=rating_info["sector"],
            )

        # selezione archi con k-NN su pesi decorrelazione
        k_neighbors = min(5, len(tickers) - 1)
        edges_to_keep = set()
        weights_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for i, src in enumerate(tickers):
            neighbor_stats: List[Tuple[float, str]] = []
            for j, dst in enumerate(tickers):
                if i == j:
                    continue
                key = tuple(sorted((src, dst)))
                corr = correlations.get(key)
                if corr is None:
                    continue
                weight = 1.0 - abs(corr)
                weights_cache[key] = (weight, corr)
                neighbor_stats.append((weight, dst))
            neighbor_stats.sort(reverse=True)
            for weight, dst in neighbor_stats[:k_neighbors]:
                if weight < self._min_weight_threshold:
                    continue
                key = tuple(sorted((src, dst)))
                edges_to_keep.add(key)

        for (src, dst) in edges_to_keep:
            weight, corr = weights_cache[(src, dst)]
            graph.add_edge(
                src,
                dst,
                weight=weight,
                correlation=corr,
                abs_correlation=abs(corr),
                sign=1 if corr >= 0 else -1,
            )

        if graph.number_of_edges() == 0:
            raise ValueError(
                "Il grafo risultante è vuoto: prova a ridurre K o modificare l'universo"
            )

        self._graph = graph
        self._current_agency = agency
        self._last_window = (self._selected_dates[0], self._selected_dates[-1])

    def getAllNodes(self) -> List[str]:
        graph = self._require_graph()
        return sorted(graph.nodes)

    def getGraphDetails(self) -> Tuple[int, int]:
        graph = self._require_graph()
        return graph.number_of_nodes(), graph.number_of_edges()

    def getCammino(self, start_node: str) -> List[str]:
        graph = self._require_graph()
        if start_node not in graph:
            raise ValueError("Nodo non presente nel grafo")

        best_path: List[str] = [start_node]
        best_weight = float("-inf")
        max_len = min(self._max_path_length, graph.number_of_nodes())

        def dfs(path: List[str], weight_sum: float) -> None:
            nonlocal best_path, best_weight
            if len(path) > 1 and weight_sum > best_weight:
                best_weight = weight_sum
                best_path = list(path)
            if len(path) == max_len:
                return
            last = path[-1]
            for neighbor, data in graph[last].items():
                if neighbor in path:
                    continue
                dfs(path + [neighbor], weight_sum + data["weight"])

        dfs([start_node], 0.0)
        return best_path

    def getBestPath(self, start_node: str) -> Tuple[List[str], float]:
        graph = self._require_graph()
        if start_node not in graph:
            raise ValueError("Nodo non presente nel grafo")

        max_len = min(self._max_path_length, graph.number_of_nodes())
        best_path: List[str] = [start_node]
        best_score = float("-inf")
        best_weight_sum = 0.0

        def dfs(
            path: List[str],
            weight_sum: float,
            rating_sum: float,
            edge_count: int,
        ) -> None:
            nonlocal best_path, best_score, best_weight_sum
            if edge_count > 0:
                avg_rating = rating_sum / len(path)
                normalized_rating = avg_rating / self._max_rating_score
                decorrelation = weight_sum / edge_count
                score = (
                    self._rating_weight * normalized_rating
                    + (1 - self._rating_weight) * decorrelation
                )
                if score > best_score:
                    best_score = score
                    best_path = list(path)
                    best_weight_sum = weight_sum
            if len(path) == max_len:
                return
            last = path[-1]
            for neighbor, data in graph[last].items():
                if neighbor in path:
                    continue
                dfs(
                    path + [neighbor],
                    weight_sum + data["weight"],
                    rating_sum + graph.nodes[neighbor]["rating_score"],
                    edge_count + 1,
                )

        start_rating = graph.nodes[start_node]["rating_score"]
        dfs([start_node], 0.0, start_rating, 0)

        if best_score == float("-inf"):
            # nessun percorso con almeno un arco
            return [start_node], 0.0
        return best_path, round(best_weight_sum, 4)

    # ------------------------------------------------------------------
    # supporto
    def _require_graph(self) -> nx.Graph:
        if self._graph is None:
            raise ValueError("Il grafo non è stato ancora creato")
        return self._graph

    def _build_return_matrix(
        self, agency: str, k_days: int
    ) -> Dict[str, List[float]]:
        if len(self._sorted_dates) < k_days:
            raise ValueError("Numero di giorni non sufficiente")
        selected_dates = self._sorted_dates[-k_days:]
        self._selected_dates = selected_dates

        returns: Dict[str, List[float]] = {}
        for ticker, _ in self._ratings[agency].items():
            prices = self._price_series.get(ticker)
            if not prices:
                continue
            try:
                series = [prices[date] for date in selected_dates]
            except KeyError:
                continue
            ticker_returns = []
            for prev, curr in zip(series[:-1], series[1:]):
                if prev <= 0 or curr <= 0:
                    ticker_returns = []
                    break
                ticker_returns.append(math.log(curr / prev))
            if len(ticker_returns) == len(selected_dates) - 1 and ticker_returns:
                returns[ticker] = ticker_returns
        return returns

    @staticmethod
    def _compute_correlations(
        returns: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], float]:
        tickers = list(returns.keys())
        correlations: Dict[Tuple[str, str], float] = {}
        for i, src in enumerate(tickers):
            for j in range(i + 1, len(tickers)):
                dst = tickers[j]
                corr = Model._pearson(returns[src], returns[dst])
                if corr is None:
                    continue
                correlations[(src, dst)] = corr
        return correlations

    @staticmethod
    def _pearson(x: Iterable[float], y: Iterable[float]) -> Optional[float]:
        x_list = list(x)
        y_list = list(y)
        if len(x_list) != len(y_list) or len(x_list) == 0:
            return None
        n = len(x_list)
        mean_x = sum(x_list) / n
        mean_y = sum(y_list) / n
        cov = 0.0
        var_x = 0.0
        var_y = 0.0
        for xi, yi in zip(x_list, y_list):
            dx = xi - mean_x
            dy = yi - mean_y
            cov += dx * dy
            var_x += dx * dx
            var_y += dy * dy
        if var_x <= 0 or var_y <= 0:
            return None
        corr = cov / math.sqrt(var_x * var_y)
        return max(min(corr, 1.0), -1.0)
