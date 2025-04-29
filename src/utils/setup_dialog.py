#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/setup_dialog.py

import os
import platform
import sys
import webbrowser
from typing import Dict, Any, Optional, Callable

# Importujemy tkinter, ale obsługujemy przypadek, gdy nie jest dostępny
try:
    if platform.system() == "Windows":
        import tkinter as tk
        from tkinter import ttk
        from tkinter import messagebox

        TKINTER_AVAILABLE = True
    else:
        # Na Linux/macOS tkinter nie zawsze jest zainstalowany
        import tkinter as tk
        from tkinter import ttk
        from tkinter import messagebox

        TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Importujemy nasz moduł wykrywania motywu
from src.utils.theme_utils import detect_system_theme

# Próbujemy zaimportować sv_ttk dla motywu Fluent Design
try:
    import sv_ttk

    SV_TTK_AVAILABLE = True
except ImportError:
    SV_TTK_AVAILABLE = False


class SetupDialog:
    """
    Dialog informujący o brakujących komponentach systemu
    i wyświetlający instrukcje instalacji.
    """

    def __init__(
            self,
            results: Dict[str, Any],
            on_continue: Optional[Callable] = None,
            on_exit: Optional[Callable] = None,
            logger=None
    ):
        """
        Inicjalizacja dialogu.

        Args:
            results (Dict[str, Any]): Wyniki sprawdzenia systemu
            on_continue (Callable, optional): Funkcja wywoływana po kliknięciu "Kontynuuj"
            on_exit (Callable, optional): Funkcja wywoływana po kliknięciu "Wyjdź"
            logger: Opcjonalny logger do zapisywania komunikatów
        """
        self.results = results
        self.on_continue = on_continue
        self.on_exit = on_exit
        self.logger = logger

        self.system = platform.system()
        self.root = None

        # Wykrywamy motyw systemu
        self.system_theme = detect_system_theme()
        if self.logger:
            self.logger.debug("SetupDialog", f"Wykryty motyw systemowy: {self.system_theme}")

    def show(self):
        """
        Wyświetla dialog z informacjami o brakujących komponentach.
        """
        if not TKINTER_AVAILABLE:
            self._show_console_message()
            return

        try:
            self.root = tk.Tk()
            self.root.title("Stick Figure Webcam - Konfiguracja")

            # Początkowy rozmiar - będzie dostosowany później
            self.root.geometry("800x600")
            self.root.minsize(600, 400)

            # Zastosowanie motywu zgodnego z systemem
            self._apply_theme()

            # Ikona aplikacji
            try:
                icon_path = self._get_resource_path("icon.ico")
                if os.path.exists(icon_path):
                    self.root.iconbitmap(icon_path)
            except Exception:
                pass  # Ignorujemy błędy ikony

            # Tworzenie zawartości
            self._create_widgets()

            # Dostosuj wielkość okna do zawartości
            self._adjust_window_size()

            self.root.mainloop()

        except Exception as e:
            if self.logger:
                self.logger.error("SetupDialog", f"Błąd wyświetlania dialogu: {str(e)}")
            self._show_console_message()

    def _apply_theme(self):
        """
        Zastosowanie motywu zgodnego z systemem operacyjnym.
        """
        # Próbujemy użyć Sun Valley jeśli jest dostępny
        if SV_TTK_AVAILABLE:
            try:
                # Ustawienie motywu Sun Valley (Fluent Design)
                sv_ttk.set_theme(self.system_theme)
                if self.logger:
                    self.logger.debug("SetupDialog", f"Zastosowano motyw Sun Valley {self.system_theme}")
                return
            except Exception as e:
                if self.logger:
                    self.logger.warning("SetupDialog", f"Nie udało się zastosować motywu Sun Valley: {str(e)}")

        # Jeśli nie udało się użyć Sun Valley, stosujemy podstawowe style Tkinter
        try:
            style = ttk.Style()

            if self.system_theme == "dark":
                # Dla ciemnego motywu ustawiamy kolory tła i tekstu
                bg_color = "#1e1e1e"
                fg_color = "#ffffff"

                # Próba ustawienia ciemnego motywu
                self.root.configure(bg=bg_color)

                # Ustawienie podstawowych stylów
                style.configure("TFrame", background=bg_color)
                style.configure("TLabel", foreground=fg_color, background=bg_color)
                style.configure("TLabelframe", foreground=fg_color, background=bg_color)
                style.configure("TLabelframe.Label", foreground=fg_color, background=bg_color)
                style.configure("TButton", foreground=fg_color, background=bg_color)

                if self.logger:
                    self.logger.debug("SetupDialog", "Zastosowano własny ciemny motyw")
            else:
                # Jasny motyw - domyślne style
                if self.logger:
                    self.logger.debug("SetupDialog", "Używanie domyślnego jasnego motywu")

        except Exception as e:
            if self.logger:
                self.logger.warning("SetupDialog", f"Nie udało się zastosować podstawowych stylów: {str(e)}")

    def _adjust_window_size(self):
        """
        Dostosowuje wielkość okna do zawartości i rozmiaru ekranu.
        """
        # Aktualizacja widgetów, aby uzyskać ich faktyczne rozmiary
        self.root.update_idletasks()

        # Pobierz preferowane rozmiary
        required_width = self.root.winfo_reqwidth()
        required_height = self.root.winfo_reqheight()

        # Dodaj umiarkowany margines
        width = required_width + 100
        height = required_height + 80

        # Ogranicz do maksymalnie 90% rozmiaru ekranu
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        max_width = int(screen_width * 0.9)
        max_height = int(screen_height * 0.9)

        # Minimalna szerokość, aby wszystkie elementy były widoczne
        min_width = 900
        min_height = 600  # Zmniejszona z 800

        # Zastosuj minimalną szerokość, ale nie większą niż maksymalna
        width = max(min(width, max_width), min_width)
        height = max(min(height, max_height), min_height)

        # Ustaw nowy rozmiar
        self.root.geometry(f"{width}x{height}")

        # Wycentruj okno
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        self.root.geometry(f"{width}x{height}+{x}+{y}")

        if self.logger:
            self.logger.debug(
                "SetupDialog",
                f"Dostosowano rozmiar okna do {width}x{height}"
            )

    def _create_widgets(self):
        """
        Tworzy widgety dla dialogu.
        """
        # Konfiguracja stylów w zależności od dostępności sv_ttk
        if not SV_TTK_AVAILABLE:
            style = ttk.Style()
            # Standardowe style Tk
            style.configure("TButton", padding=6, relief="flat", font=("Segoe UI", 10))
            style.configure("TLabel", font=("Segoe UI", 10))
            style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
            style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
            style.configure("Missing.TLabel", foreground="red")
            style.configure("OK.TLabel", foreground="green")
        else:
            # Przy użyciu sv_ttk tylko dostosowujemy czcionki i kolory
            style = ttk.Style()
            style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
            style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))

            # Wzmocnione kolory dla lepszego kontrastu
            if self.system_theme == "dark":
                style.configure("Missing.TLabel", foreground="#ff8080")  # Jaśniejsza czerwień w ciemnym motywie
                style.configure("OK.TLabel", foreground="#80ff80")  # Jaśniejsza zieleń w ciemnym motywie
                # Jasny niebieski dla linków w ciemnym motywie
                style.configure("Link.TLabel", foreground="#40a9ff", font=("Segoe UI", 10, "underline"))
            else:
                style.configure("Missing.TLabel", foreground="#e03131")  # Ciemna czerwień w jasnym motywie
                style.configure("OK.TLabel", foreground="#2b8a3e")  # Ciemna zieleń w jasnym motywie
                # Ciemny niebieski dla linków w jasnym motywie
                style.configure("Link.TLabel", foreground="#0366d6", font=("Segoe UI", 10, "underline"))

        # Główny kontener z paddingiem
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Tytuł
        title_label = ttk.Label(
            main_frame,
            text="Stick Figure Webcam - Wymagania Systemowe",
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 10))

        # Opis
        description = (
            "Aplikacja wymaga kilku komponentów do poprawnego działania. "
            "Poniżej znajduje się lista sprawdzonych elementów oraz instrukcje "
            "instalacji brakujących składników."
        )
        desc_label = ttk.Label(main_frame, text=description, wraplength=800)
        desc_label.pack(pady=(0, 10), fill=tk.X)

        # Ramka dla wyników sprawdzenia
        results_frame = ttk.LabelFrame(main_frame, text="Wyniki sprawdzenia", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Tworzymy frame dla zawartości zamiast canvas ze scrollbarem
        content_frame = ttk.Frame(results_frame, padding=5)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Wyświetlenie statusu każdego komponentu
        self._show_component_status(content_frame, "Kamera", "camera")
        self._show_component_status(content_frame, "Wirtualna kamera", "virtual_camera")
        self._show_component_status(content_frame, "MediaPipe", "mediapipe")

        if self.system == "Windows":
            self._show_component_status(content_frame, "OBS Studio", "obs")
        elif self.system == "Linux":
            self._show_component_status(content_frame, "v4l2loopback", "v4l2loopback")

        # Instrukcje instalacji
        if not self.results["all_met"] and self.results["instructions"]:
            instructions_label = ttk.Label(
                content_frame,
                text="Instrukcje instalacji:",
                style="Header.TLabel"
            )
            instructions_label.pack(anchor="w", pady=(20, 10))

            self._show_instructions(content_frame)

        # Przyciski
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        # Zawsze wyświetlaj obydwa przyciski
        continue_button = ttk.Button(
            button_frame,
            text="Kontynuuj mimo to",
            command=self._on_continue
        )

        exit_button = ttk.Button(
            button_frame,
            text="Zamknij aplikację",
            command=self._on_exit
        )

        # Przycisk wyjścia po lewej, kontynuacji po prawej
        exit_button.pack(side=tk.LEFT, padx=5)
        continue_button.pack(side=tk.RIGHT, padx=5)

        # Dodatkowa informacja
        if not self.results["all_met"]:
            warning_label = ttk.Label(
                main_frame,
                text="Uwaga: Aplikacja może nie działać poprawnie bez wymaganych komponentów!",
                style="Missing.TLabel"
            )
            warning_label.pack(pady=(10, 0))

    def _show_component_status(self, parent, display_name, component_key):
        """
        Wyświetla status sprawdzenia komponentu.

        Args:
            parent: Widget rodzica
            display_name (str): Nazwa wyświetlana komponentu
            component_key (str): Klucz komponentu w wynikach
        """
        if component_key not in self.results["results"]:
            return

        result = self.results["results"][component_key]

        component_frame = ttk.Frame(parent)
        component_frame.pack(fill=tk.X, pady=5)

        # Ustalanie stylu i tekstu statusu
        if result["status"] is True:
            status_text = "✓ Zainstalowany"

            # Bezpośrednie ustawienie koloru zamiast polegania na stylach
            if self.system_theme == "dark":
                status_color = "#80ff80"  # Jaśniejsza zieleń dla ciemnego motywu
            else:
                status_color = "#2b8a3e"  # Ciemna zieleń dla jasnego motywu
        elif result["status"] is False:
            status_text = "✗ Brak"

            # Bezpośrednie ustawienie koloru zamiast polegania na stylach
            if self.system_theme == "dark":
                status_color = "#ff8080"  # Jaśniejsza czerwień dla ciemnego motywu
            else:
                status_color = "#e03131"  # Ciemna czerwień dla jasnego motywu
        else:
            status_text = "- Nie sprawdzono"
            status_color = None

        name_label = ttk.Label(component_frame, text=f"{display_name}:", width=20)
        name_label.pack(side=tk.LEFT)

        # Bezpośrednie ustawienie koloru dla etykiety statusu
        status_label = ttk.Label(component_frame, text=status_text, foreground=status_color)
        status_label.pack(side=tk.LEFT, padx=10)

        # Dodatkowy opis - bezpośrednie kolorowanie tekstu
        if "message" in result and result["message"]:
            message_color = None

            # Określenie koloru wiadomości na podstawie statusu i treści
            if result["status"] is True:
                # Zielony dla pozytywnych komunikatów
                message_color = "#80ff80" if self.system_theme == "dark" else "#2b8a3e"
            elif result["status"] is False and (
                    "błąd" in result["message"].lower() or "nie " in result["message"].lower()):
                # Czerwony dla negatywnych komunikatów
                message_color = "#ff8080" if self.system_theme == "dark" else "#e03131"

            # Tworzenie etykiety z bezpośrednim ustawieniem koloru
            desc_label = ttk.Label(
                component_frame,
                text=result["message"],
                foreground=message_color,
                wraplength=600
            )
            desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _show_instructions(self, parent):
        """
        Wyświetla instrukcje instalacji brakujących komponentów.

        Args:
            parent: Widget rodzica
        """
        for component, instructions in self.results["instructions"].items():
            # Określenie nazwy wyświetlanej
            display_name = {
                "camera": "Kamera",
                "virtual_camera": "Wirtualna kamera",
                "mediapipe": "MediaPipe",
                "obs": "OBS Studio",
                "v4l2loopback": "v4l2loopback"
            }.get(component, component)

            # Ramka dla komponentu
            component_frame = ttk.LabelFrame(parent, text=display_name, padding=10)
            component_frame.pack(fill=tk.X, pady=5)

            # Wyświetlenie instrukcji
            for i, instruction in enumerate(instructions):
                instruction_frame = ttk.Frame(component_frame)
                instruction_frame.pack(fill=tk.X, pady=2)

                # Sprawdzenie, czy to link
                if "http" in instruction:
                    step_label = ttk.Label(instruction_frame, text=f"{i + 1}. ")
                    step_label.pack(side=tk.LEFT, anchor="nw")

                    # Wyodrębnienie tekstu i linku
                    parts = instruction.split(": ", 1)
                    if len(parts) > 1:
                        text, link = parts
                        text += ": "
                    else:
                        text = ""
                        link = instruction

                    if text:
                        text_label = ttk.Label(instruction_frame, text=text)
                        text_label.pack(side=tk.LEFT, anchor="nw")

                    # Bardzo jasny kolor linku dla ciemnego motywu dla lepszego kontrastu
                    link_color = "#00aaff" if self.system_theme == "dark" else "#0366d6"

                    link_label = ttk.Label(
                        instruction_frame,
                        text=link,
                        foreground=link_color,
                        font=("Segoe UI", 10, "underline"),
                        cursor="hand2"
                    )
                    link_label.pack(side=tk.LEFT, anchor="nw")

                    # Obsługa kliknięcia linku
                    link_label.bind(
                        "<Button-1>",
                        lambda e, url=link: webbrowser.open_new(url)
                    )
                else:
                    instr_label = ttk.Label(
                        instruction_frame,
                        text=f"{i + 1}. {instruction}",
                        wraplength=800  # Zwiększona szerokość zawijania
                    )
                    instr_label.pack(anchor="w")

    def _on_continue(self):
        """
        Obsługuje kliknięcie przycisku "Kontynuuj".
        """
        if self.logger:
            self.logger.info(
                "SetupDialog",
                "Użytkownik zdecydował się kontynuować mimo brakujących komponentów"
            )

        if self.on_continue:
            self.on_continue()

        if self.root:
            self.root.destroy()

    def _on_exit(self):
        """
        Obsługuje kliknięcie przycisku "Wyjdź".
        """
        if self.logger:
            self.logger.info("SetupDialog", "Użytkownik zdecydował się zamknąć aplikację")

        if self.on_exit:
            self.on_exit()

        if self.root:
            self.root.destroy()
            sys.exit(0)

    def _get_resource_path(self, relative_path):
        """
        Zwraca ścieżkę do zasobów, działającą zarówno w trybie deweloperskim jak i w pakiecie.

        Args:
            relative_path (str): Ścieżka względna do zasobu

        Returns:
            str: Pełna ścieżka do zasobu
        """
        base_path = getattr(sys, '_MEIPASS',
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(base_path, "resources", relative_path)

    def _show_console_message(self):
        """
        Wyświetla informacje w konsoli, gdy nie można utworzyć GUI.
        """
        print("\n" + "=" * 80)
        print("STICK FIGURE WEBCAM - SPRAWDZENIE WYMAGAŃ SYSTEMOWYCH")
        print("=" * 80)

        # Wyświetl status każdego komponentu
        component_names = {
            "camera": "Kamera",
            "virtual_camera": "Wirtualna kamera",
            "mediapipe": "MediaPipe",
            "obs": "OBS Studio",
            "v4l2loopback": "v4l2loopback"
        }

        for component, result in self.results["results"].items():
            if component in component_names:
                display_name = component_names[component]

                # Status
                if result["status"] is True:
                    status = "[ZAINSTALOWANY]"
                elif result["status"] is False:
                    status = "[BRAK]"
                else:
                    status = "[NIE SPRAWDZONO]"

                print(f"{display_name:20} {status:15} {result.get('message', '')}")

        print("\n")

        # Wyświetl instrukcje, jeśli są brakujące komponenty
        if not self.results["all_met"]:
            print("INSTRUKCJE INSTALACJI BRAKUJĄCYCH KOMPONENTÓW:")
            print("-" * 80)

            for component, instructions in self.results["instructions"].items():
                if component in component_names:
                    display_name = component_names[component]
                    print(f"\n{display_name}:")

                    for i, instruction in enumerate(instructions):
                        print(f"  {i + 1}. {instruction}")

            print("\n")
            print("UWAGA: Aplikacja może nie działać poprawnie bez wymaganych komponentów!")

        print("=" * 80)

        # Pytanie o kontynuację
        if not self.results["all_met"]:
            response = input("Czy chcesz kontynuować mimo to? (t/n): ").lower()

            if response == 't' or response == 'tak':
                if self.on_continue:
                    self.on_continue()
            else:
                if self.on_exit:
                    self.on_exit()
                sys.exit(0)
        else:
            if self.on_continue:
                self.on_continue()


# Funkcja pomocnicza do użycia w main.py
def show_setup_dialog(
        results: Dict[str, Any],
        on_continue: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        logger=None
):
    """
    Wyświetla dialog konfiguracyjny dla brakujących komponentów.

    Args:
        results (Dict[str, Any]): Wyniki sprawdzenia systemu
        on_continue (Callable, optional): Funkcja wywoływana po kliknięciu "Kontynuuj"
        on_exit (Callable, optional): Funkcja wywoływana po kliknięciu "Wyjdź"
        logger: Opcjonalny logger do zapisywania komunikatów
    """
    # Jeśli wszystkie wymagania są spełnione, nie ma potrzeby wyświetlania dialogu
    if results["all_met"]:
        if logger:
            logger.info("SetupDialog", "Wszystkie wymagania systemowe są spełnione")

        if on_continue:
            on_continue()
        return

    # Tworzenie i wyświetlanie dialogu
    dialog = SetupDialog(results, on_continue, on_exit, logger)
    dialog.show()
