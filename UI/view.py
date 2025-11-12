import flet as ft


class View(ft.UserControl):
    def __init__(self, page: ft.Page):
        super().__init__()
        # page stuff
        self._page = page
        self._page.title = "TdP Lab 14 - simulazione esame"
        self._page.horizontal_alignment = 'CENTER'
        self._page.theme_mode = ft.ThemeMode.LIGHT
        self._page.window_height = 800
        page.window_center()
        # controller (it is not initialized. Must be initialized in the main, after the controller is created)
        self._controller = None
        # graphical elements
        self._title = None
        self._txt_name = None
        self._txt_result = None

    def load_interface(self):
        # 1) Switch tema
        self.__theme_switch = ft.Switch(
            label="Light theme",
            value=False,  # False = light, True = dark
            on_change=self.theme_changed
        )

        # Titolo in pagina (Text)
        self._title = ft.Text("TdP Lab 14 – simulazione esame", color="blue", size=24)

        # Header con switch e titolo
        header = ft.Row(
            controls=[
                ft.Container(self.__theme_switch, padding=10),
                ft.Container(self._title, expand=True,
                             alignment= ft.alignment.top_center),
            ],
            alignment=ft.MainAxisAlignment.CENTER
        )


        self._ddStore = ft.Dropdown(label="Store")
        self._txtIntK = ft.TextField(label="Numero giorni massimo K")
        self._btnCreaGrafo = ft.ElevatedButton(text="Crea Grafo", on_click=self._controller.handleCreaGrafo)
        cont = ft.Container(self._ddStore, width=250, alignment=ft.alignment.top_left)
        row1 = ft.Row([cont, self._txtIntK, self._btnCreaGrafo], alignment=ft.MainAxisAlignment.CENTER,
                      vertical_alignment=ft.CrossAxisAlignment.END)

        self._controller.fillDDStore()

        self._btnCerca = ft.ElevatedButton(text="Cerca Percorso Massimo",
                                           on_click=self._controller.handleCerca)

        self._ddNode = ft.Dropdown(label="Node")
        cont2 = ft.Container(self._ddNode, width=250, alignment=ft.alignment.top_left)


        self._btnRicorsione = ft.ElevatedButton(text="Ricorsione",
                                           on_click=self._controller.handleRicorsione)

        row1 = ft.Row([cont,
                       self._txtIntK,
                       self._btnCreaGrafo,
                       ],
                      alignment=ft.MainAxisAlignment.CENTER,
                      vertical_alignment=ft.CrossAxisAlignment.END)
        row2 = ft.Row([cont2,
                       ft.Container(self._btnCerca, width=250)
                       ], alignment=ft.MainAxisAlignment.CENTER)

        row3 = ft.Row([ft.Container(self._btnRicorsione, width=250)
                       ],
                      alignment=ft.MainAxisAlignment.CENTER)
        self._page.add(header,
                       row1,
                       row2,
                       row3,
                       )

        self.txt_result = ft.ListView(expand=1, spacing=10, padding=20, auto_scroll=True)
        self._page.controls.append(self.txt_result)
        self._page.update()

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
    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, controller):
        self._controller = controller

    def set_controller(self, controller):
        self._controller = controller

    def update_page(self):
        self._page.update()