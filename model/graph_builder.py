from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


class GraphBuilder:
    """
    Funzioni per:
    - costruire matrici di distanza da una matrice di correlazione,
    - applicare filtri (soglia, k-NN),
    - costruire il grafo NetworkX e la matrice di adiacenza.
    """

    @staticmethod
    def build_distance_matrix(rho: pd.DataFrame, signed: bool = False) -> pd.DataFrame:
        """
        Costruisce una matrice delle distanze a partire da una matrice di correlazione rho.

        - se signed=False: d_ij = 1 - |rho_ij|
        - se signed=True:  d_ij = (1 - rho_ij) / 2

        La diagonale viene posta a 0.
        """
        if rho is None or rho.empty:
            raise ValueError("rho è vuota.")

        if signed:
            d_values = (1.0 - rho.values) / 2.0
        else:
            d_values = 1.0 - np.abs(rho.values)

        d = pd.DataFrame(d_values, index=rho.index, columns=rho.columns)
        np.fill_diagonal(d.values, 0.0)
        return d

    @staticmethod
    def threshold_filter(rho: pd.DataFrame, tau: float) -> pd.DataFrame:
        """
        (Utility opzionale, usata solo in esperimenti)

        Applica un filtro a soglia alla matrice di correlazione:
        - mantiene i valori con |rho_ij| >= tau
        - pone a 0 gli altri
        """
        if rho is None or rho.empty:
            raise ValueError("rho è vuota.")
        if not (0.0 <= tau <= 1.0):
            raise ValueError("tau deve essere in [0,1].")

        rho_f = rho.copy()
        mask = rho_f.abs() < tau
        rho_f[mask] = 0.0
        np.fill_diagonal(rho_f.values, 0.0)
        return rho_f

    @staticmethod
    def knn_filter(
        distance: pd.DataFrame,
        k: int,
        symmetric: bool = True
    ) -> pd.DataFrame:
        """
        Applica un filtro k-NN sulla matrice delle distanze.

        Restituisce una matrice D_knn dove:
        - per ciascuna riga i, solo le k distanze più piccole restano finite,
        - tutte le altre voci sono +inf.
        - se symmetric=True, si rende la matrice simmetrica.
        """
        if distance is None or distance.empty:
            raise ValueError("distance è vuota.")
        if k <= 0:
            raise ValueError("k deve essere > 0.")

        n = distance.shape[0]
        if k >= n:
            d_knn = distance.copy()
            np.fill_diagonal(d_knn.values, 0.0)
            return d_knn

        inf = np.inf
        d_knn = pd.DataFrame(
            inf,
            index=distance.index,
            columns=distance.columns
        )

        for i, row_label in enumerate(distance.index):
            row = distance.loc[row_label].copy()
            row_no_diag = row.drop(labels=row_label)
            k_smallest = row_no_diag.nsmallest(k)

            for col_label, val in k_smallest.items():
                d_knn.at[row_label, col_label] = val

        if symmetric:
            vals = d_knn.values
            for i in range(n):
                for j in range(i + 1, n):
                    vij = vals[i, j]
                    vji = vals[j, i]
                    v = min(vij, vji)
                    vals[i, j] = v
                    vals[j, i] = v

            d_knn = pd.DataFrame(vals, index=distance.index, columns=distance.columns)

        np.fill_diagonal(d_knn.values, 0.0)
        return d_knn


    @staticmethod
    def build_filtered_graph(
        rho: pd.DataFrame,
        tau: float | None = None,
        k: int | None = None,
        signed: bool = False,
    ) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
        """
        A partire da una matrice di correlazione rho:
        - costruisce la matrice delle distanze (eventualmente signed),
        - applica filtro a soglia (tau) e k-NN (k),
        - costruisce la matrice di adiacenza adj (|rho_ij| dove esiste arco, 0 altrimenti),
        - costruisce il grafo NetworkX con:
              weight = |rho_ij|, corr = rho_ij.

        Restituisce: (G, adj, dist_knn).
        """
        if rho is None or rho.empty:
            raise ValueError("rho è vuota.")

        # 1) matrice distanze base
        dist = GraphBuilder.build_distance_matrix(rho, signed=signed)

        # 2) modulo della correlazione, con diagonale a 0
        abs_rho = rho.abs().copy()
        np.fill_diagonal(abs_rho.values, 0.0)

        # 3) filtro a soglia sulla distanza (dove |rho| < tau → distanza = +inf)
        if tau is not None:
            if not (0.0 <= tau <= 1.0):
                raise ValueError("tau deve essere in [0,1].")
            mask_below = abs_rho.values < tau
            dist.values[mask_below] = np.inf

        # 4) filtro k-NN (se richiesto)
        if k is not None:
            dist_knn = GraphBuilder.knn_filter(dist, k=k, symmetric=True)
        else:
            dist_knn = dist

        # 5) matrice di adiacenza: |rho_ij| se dist_ij finita, 0 altrove
        adj = abs_rho.copy()
        mask_no_edge = ~np.isfinite(dist_knn.values)
        adj.values[mask_no_edge] = 0.0
        np.fill_diagonal(adj.values, 0.0)

        # 6) costruzione grafo NetworkX
        G = nx.Graph()
        for t in adj.index:
            G.add_node(t)

        cols = list(adj.columns)
        for i, ti in enumerate(adj.index):
            row_vals = adj.iloc[i].values
            for j in range(i + 1, len(cols)):
                w = row_vals[j]
                if w <= 0.0:
                    continue
                tj = cols[j]
                corr_val = rho.loc[ti, tj]
                G.add_edge(ti, tj, weight=w, corr=corr_val)

        return G, adj, dist_knn


if __name__ == "__main__":
    # Mini test indipendente con una matrice fittizia
    import traceback

    data = {
        "A": [1.0, 0.5, -0.2],
        "B": [0.5, 1.0, 0.3],
        "C": [-0.2, 0.3, 1.0],
    }
    rho_test = pd.DataFrame(data, index=["A", "B", "C"])

    print("=== TEST GRAPH_BUILDER (OOP) ===")
    try:
        G, adj, dist_knn = GraphBuilder.build_filtered_graph(
            rho_test,
            tau=0.25,
            k=1,
            signed=False,
        )
        print("Adj:\n", adj)
        print("Distanza (k-NN):\n", dist_knn)
        print("Nodi grafo:", G.nodes())
        print("Archi grafo (con attributi):")
        for u, v, attr in G.edges(data=True):
            print(f"{u}-{v}: {attr}")

    except Exception:
        print("TEST GRAPH_BUILDER FALLITO:")
        traceback.print_exc()
    else:
        print("TEST GRAPH_BUILDER OK.")
