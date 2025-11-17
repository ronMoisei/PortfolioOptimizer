import flet as ft


class View(ft.UserControl):
    def __init__(self, page: ft.Page):
        super().__init__()
        # page stuff
        self._page = page
        self._page.title = "Portfolio Optimizer"
        self._page.horizontal_alignment = "CENTER"
        self._page.theme_mode = ft.ThemeMode.LIGHT
        self._page.window_height = 800
        self._page.window_width = 1100
        self._page.window_center()

        # controller (verrà settato dal main)
        self._controller = None

        # graphical elements
        self._title = None
        self.__theme_switch = None

        # parametri ottimizzazione
        self._txtK = None
        self._ddMaxUnrated = None
        self._btnOptimize = None

        # area risultati
        self.txt_result = None

    # ------------------------------------------------------------------ #
    # COSTRUZIONE INTERFACCIA
    # ------------------------------------------------------------------ #
    def load_interface(self):
        # Switch tema
        self.__theme_switch = ft.Switch(
            label="Light theme",
            value=False,  # False = light, True = dark
            on_change=self.theme_changed,
        )

        # Titolo
        self._title = ft.Text("Portfolio Optimizer", color="blue", size=24)

        # Header con switch e titolo
        header = ft.Row(
            controls=[
                ft.Container(self.__theme_switch, padding=10),
                ft.Container(self._title, expand=True, alignment=ft.alignment.top_center),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )

        # ------------------- Parametri ottimizzazione ------------------- #

        # Numero titoli K
        self._txtK = ft.TextField(
            label="Numero titoli (K)",
            hint_text="Es: 4",
            width=150,
        )

        # Quota massima unrated
        self._ddMaxUnrated = ft.Dropdown(
            label="Quota max titoli unrated",
            width=220,
            options=[
                ft.dropdown.Option("0.0", "0% (solo rated)"),
                ft.dropdown.Option("0.2", "20%"),
                ft.dropdown.Option("0.5", "50%"),
                ft.dropdown.Option("1.0", "100% (nessun vincolo)"),
            ],
            value="0.2",
        )

        # Bottone ottimizzazione
        self._btnOptimize = ft.ElevatedButton(
            text="Ottimizza portafoglio",
            on_click=self._controller.handle_optimize if self._controller else None,
        )

        params_row = ft.Row(
            [
                ft.Container(self._txtK, alignment=ft.alignment.top_left),
                ft.Container(self._ddMaxUnrated, alignment=ft.alignment.top_left),
                ft.Container(self._btnOptimize, alignment=ft.alignment.top_left),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )

        # ------------------- Area risultati ------------------- #

        self.txt_result = ft.ListView(
            expand=1,
            spacing=6,
            padding=20,
            auto_scroll=True,
        )

        # Layout pagina
        self._page.add(
            header,
            ft.Divider(),
            params_row,
            ft.Divider(),
            self.txt_result,
        )
        self._page.update()

    # ------------------------------------------------------------------ #
    # CAMBIO TEMA
    # ------------------------------------------------------------------ #
    def theme_changed(self, e: ft.ControlEvent):
        # inverte tema
        self._page.theme_mode = (
            ft.ThemeMode.DARK
            if self._page.theme_mode == ft.ThemeMode.LIGHT
            else ft.ThemeMode.LIGHT
        )
        # aggiorna label
        self.__theme_switch.label = (
            "Light theme"
            if self._page.theme_mode == ft.ThemeMode.LIGHT
            else "Dark theme"
        )
        self._page.update()

    # ------------------------------------------------------------------ #
    # CONTROLLER PROPERTY
    # ------------------------------------------------------------------ #
    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, controller):
        self._controller = controller
        # collega il bottone al nuovo controller (se l'interfaccia è già stata creata)
        if self._btnOptimize is not None:
            self._btnOptimize.on_click = self._controller.handle_optimize
        self._page.update()

    def set_controller(self, controller):
        self.controller = controller

    # ------------------------------------------------------------------ #
    # AGGIORNA PAGINA
    # ------------------------------------------------------------------ #
    def update_page(self):
        self._page.update()
