import flet as ft

class Controller:
    def __init__(self, view, model):
        self._view = view
        self._model = model

    def fillDDStore(self):
        try:
            stores = self._model.getStores()
            self._view._ddStore.options.clear()
            for store in stores:
                print(store)
                self._view._ddStore.options.append(ft.dropdown.Option(f"{store[0]}-{store[1]}"))
            self._view.update_page()
        except Exception as ex:
            self._show_error(f"Errore caricamento store: {ex}")

    def handleCreaGrafo(self, e):
        try:
            # reset area messaggi
            self._view.txt_result.controls.clear()

            if not self._view._ddStore.value:
                raise ValueError("Devi selezionare uno store.")
            k_str = self._view._txtIntK.value or ""
            if not k_str.isdigit():
                raise ValueError("K deve essere un intero positivo.")
            kint = int(k_str)

            self._model.buildGraph(self._view._ddStore.value, kint)

            allNodes = self._model.getAllNodes()
            self.fillDD(allNodes)

            Nnodes, Nedges = self._model.getGraphDetails()
            self._view.txt_result.controls.append(ft.Text("Grafo correttamente creato:"))
            self._view.txt_result.controls.append(ft.Text(f"Numero di nodi: {Nnodes}"))
            self._view.txt_result.controls.append(ft.Text(f"Numero di archi: {Nedges}"))
            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore creazione grafo: {ex}")

    def handleCerca(self, e):
        try:
            self._view.txt_result.controls.clear()
            if not self._view._ddNode.value:
                raise ValueError("Devi selezionare un nodo di partenza.")
            nodes = self._model.getCammino(self._view._ddNode.value)
            self._view.txt_result.controls.append(ft.Text(f"Nodo di partenza: {self._view._ddNode.value}"))
            for n in nodes:
                self._view.txt_result.controls.append(ft.Text(str(n)))
            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore ricerca cammino: {ex}")

    def handleRicorsione(self, e):
        try:
            self._view.txt_result.controls.clear()
            if not self._view._ddNode.value:
                raise ValueError("Devi selezionare un nodo di partenza.")
            bestpath, bestscore = self._model.getBestPath(self._view._ddNode.value)
            self._view.txt_result.controls.append(
                ft.Text(f"Trovato un cammino che parte da {self._view._ddNode.value} "
                        f"con somma dei pesi uguale a {bestscore}.")
            )
            for v in bestpath:
                self._view.txt_result.controls.append(ft.Text(str(v)))
            self._view.update_page()

        except Exception as ex:
            self._show_error(f"Errore calcolo ricorsione: {ex}")

    def fillDD(self, allNodes):
        try:
            self._view._ddNode.options.clear()
            for n in allNodes:
                self._view._ddNode.options.append(ft.dropdown.Option(n))
            self._view.update_page()
        except Exception as ex:
            self._show_error(f"Errore popolamento nodi: {ex}")

    def _show_error(self, message: str):
        """Mostra un messaggio di errore in txt_result."""
        self._view.txt_result.controls.clear()
        self._view.txt_result.controls.append(ft.Text(message, color="red"))
        self._view.update_page()
