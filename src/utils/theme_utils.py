#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/theme_utils.py

import ctypes
import os
import platform
import subprocess
from typing import Tuple, Optional


def detect_system_theme() -> str:
    """
    Wykrywa motyw systemowy (jasny/ciemny).

    Returns:
        str: "dark" lub "light"
    """
    system = platform.system()

    try:
        # Windows
        if system == "Windows":
            # Używamy Windows API do sprawdzenia motywu
            try:
                # Używając ctypes na Windows 10+
                HKEY_CURRENT_USER = -2147483647
                SUBKEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"

                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, SUBKEY)
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    winreg.CloseKey(key)
                    return "light" if value == 1 else "dark"
                except:
                    # Jeśli nie można użyć rejestru, używamy domyślnego koloru okna
                    return "dark" if ctypes.windll.user32.GetSysColor(13) < 127 else "light"
            except:
                return "light"  # Domyślnie jasny motyw na starszych wersjach Windows

        # macOS
        elif system == "Darwin":
            try:
                # Używamy polecenia systemowego Apple
                result = subprocess.run(
                    ["defaults", "read", "-g", "AppleInterfaceStyle"],
                    capture_output=True,
                    text=True
                )
                # Jeśli wynikiem jest "Dark", użytkownik ma włączony ciemny motyw
                # Jeśli polecenie zwróci błąd, użytkownik ma jasny motyw
                return "dark" if "Dark" in result.stdout else "light"
            except:
                return "light"  # Domyślnie jasny motyw

        # Linux
        elif system == "Linux":
            try:
                # Sprawdzamy GNOME
                result = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                    capture_output=True,
                    text=True
                )
                if "dark" in result.stdout.lower():
                    return "dark"
                return "light"
            except:
                # Sprawdzamy zmienną środowiskową GTK
                gtk_theme = os.environ.get("GTK_THEME", "").lower()
                if "dark" in gtk_theme:
                    return "dark"
                return "light"  # Domyślnie jasny motyw

        else:
            return "light"  # Domyślnie jasny motyw

    except Exception:
        return "light"  # W razie błędu, zakładamy jasny motyw


def get_theme_colors(theme: str = None) -> Tuple[str, str, str, str, str]:
    """
    Zwraca kolory odpowiednie dla motywu.

    Args:
        theme (str, optional): "dark", "light" lub None aby wykryć automatycznie

    Returns:
        Tuple[str, str, str, str, str]:
            - Kolor podstawowy tekstu
            - Kolor podświetlenia
            - Kolor tła
            - Kolor powodzenia (zielony)
            - Kolor błędu (czerwony)
    """
    if theme is None:
        theme = detect_system_theme()

    if theme == "dark":
        # Ciemny motyw - z jaśniejszymi kolorami dla lepszego kontrastu
        return (
            "#ffffff",  # Tekst: biały
            "#1890ff",  # Podświetlenie: jasny niebieski
            "#1e1e1e",  # Tło: ciemny szary
            "#80ff80",  # Sukces: jaskrawa zieleń
            "#ff8080"  # Błąd: jaskrawa czerwień
        )
    else:
        # Jasny motyw
        return (
            "#000000",  # Tekst: czarny
            "#0366d6",  # Podświetlenie: ciemny niebieski
            "#f0f0f0",  # Tło: jasny szary
            "#2b8a3e",  # Sukces: ciemna zieleń
            "#e03131"  # Błąd: ciemna czerwień
        )


def apply_theme_to_tkinter(root, theme: Optional[str] = None) -> bool:
    """
    Stosuje motyw do okna Tkinter.

    Args:
        root: Główne okno Tkinter
        theme (str, optional): "dark", "light" lub None aby wykryć automatycznie

    Returns:
        bool: True jeśli zastosowano motyw, False w przeciwnym razie
    """
    if theme is None:
        theme = detect_system_theme()

    try:
        # Próbujemy użyć Sun Valley Tkinter Theme (sv_ttk)
        import sv_ttk
        sv_ttk.set_theme(theme)
        return True
    except ImportError:
        # Jeśli sv_ttk nie jest dostępny, stosujemy podstawowe style
        try:
            from tkinter import ttk
            style = ttk.Style()

            text_color, accent_color, bg_color, success_color, error_color = get_theme_colors(theme)

            if theme == "dark":
                # Ciemny motyw - próba dostosowania standardowych stylów
                style.configure("TFrame", background=bg_color)
                style.configure("TLabel", foreground=text_color, background=bg_color)
                style.configure("TButton", foreground=text_color, background=accent_color)

                # Dostosowanie kolorów specjalnych
                style.configure("Success.TLabel", foreground=success_color)
                style.configure("Error.TLabel", foreground=error_color)

                # Ustawienie kolorów okna głównego
                root.configure(background=bg_color)
            else:
                # Jasny motyw - domyślne style
                pass

            return True
        except Exception:
            return False  # Nie udało się zastosować motywu
