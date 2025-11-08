#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/logging_config.py

import datetime
import os
from typing import Optional, Dict, Any


def get_log_file_path() -> str:
    """
    Generuje ścieżkę do pliku logów na podstawie aktualnej daty.

    Returns:
        str: Ścieżka do pliku logów
    """
    # Aktualny katalog projektu
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Katalog na logi
    logs_dir = os.path.join(project_dir, "logs")

    # Utworzenie katalogu, jeśli nie istnieje
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # Format nazwy pliku: stick_figure_YYYY-MM-DD_HH-MM-SS.log
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"stick_figure_{timestamp}.log"

    return os.path.join(logs_dir, log_filename)


def get_logger_config(debug: bool = False) -> Dict[str, Any]:
    """
    Zwraca konfigurację loggera.

    Args:
        debug (bool): Czy włączyć tryb debugowania

    Returns:
        Dict[str, Any]: Konfiguracja loggera
    """
    log_file = get_log_file_path()

    return {
        "log_file": log_file,
        "console_level": "DEBUG" if debug else "INFO",
        "file_level": "DEBUG",
        "timezone": "Europe/Warsaw"
    }


def setup_logger(custom_logger_class, debug: bool = False) -> Optional[Any]:
    """
    Tworzy i konfiguruje instancję loggera.

    Args:
        custom_logger_class: Klasa loggera (np. CustomLogger)
        debug (bool): Czy włączyć tryb debugowania

    Returns:
        Optional[Any]: Skonfigurowany logger lub None w przypadku błędu
    """
    try:
        logger_config = get_logger_config(debug)
        logger = custom_logger_class(**logger_config)

        logger.info(
            "LoggingConfig",
            f"Logger zainicjalizowany. Logi zapisywane do: {logger_config['log_file']}",
            log_type="CONFIG"
        )

        return logger
    except Exception as e:
        print(f"Błąd podczas inicjalizacji loggera: {str(e)}")
        # Awaryjne utworzenie loggera tylko z loggowaniem na konsolę
        try:
            fallback_logger = custom_logger_class(log_file=None, console_level="INFO")
            fallback_logger.error(
                "LoggingConfig",
                f"Błąd podczas inicjalizacji loggera: {str(e)}. Używam awaryjnego loggera bez zapisu do pliku.",
                log_type="CONFIG"
            )
            return fallback_logger
        except:
            print("Krytyczny błąd podczas inicjalizacji loggera awaryjnego.")
            return None
