#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import json
import logging
import os
import re

import pytz
from colorama import init, Fore, Back, Style

# Inicjalizacja colorama
init(autoreset=True)


class CustomLogger:
    """
    Logger z kolorowym formatowaniem dla konsoli i czystymi plikami logów.
    Przydatny do debugowania aplikacji computer vision w czasie rzeczywistym.
    """

    # Poziomy logowania
    LEVELS = {
        "TRACE": {"color": Fore.MAGENTA, "symbol": "🔬", "level": 5},
        "DEBUG": {"color": Fore.CYAN, "symbol": "🔍", "level": logging.DEBUG},
        "INFO": {"color": Fore.GREEN, "symbol": "ℹ️", "level": logging.INFO},
        "WARNING": {"color": Fore.YELLOW, "symbol": "⚠️", "level": logging.WARNING},
        "ERROR": {"color": Fore.RED, "symbol": "❌", "level": logging.ERROR},
        "CRITICAL": {"color": Fore.RED + Back.WHITE, "symbol": "🔥", "level": logging.CRITICAL},
    }

    # Specjalne typy logów dostosowane do projektu stick figure
    TYPES = {
        "CAMERA": {"color": Fore.BLUE, "symbol": "📷"},
        "POSE": {"color": Fore.MAGENTA, "symbol": "👤"},
        "DRAWING": {"color": Fore.YELLOW, "symbol": "🖌️"},
        "VIRTUAL_CAM": {"color": Fore.LIGHTBLUE_EX, "symbol": "📺"},
        "CONFIG": {"color": Fore.GREEN, "symbol": "⚙️"},
        "PERFORMANCE": {"color": Fore.CYAN, "symbol": "⏱️"},
    }

    # Dodajemy poziom TRACE do biblioteki logging
    logging.addLevelName(5, "TRACE")

    class ColoredFormatter(logging.Formatter):
        """Formatter dodający kolory dla konsoli."""

        def format(self, record):
            return record.msg

    class PlainFormatter(logging.Formatter):
        """Formatter bez kolorów dla pliku."""

        def format(self, record):
            if hasattr(record, 'plain_msg'):
                return record.plain_msg
            # Usuwamy sekwencje ANSI z wiadomości
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_escape.sub('', record.msg)

    def __init__(self, log_file=None, console_level="INFO", file_level="DEBUG", timezone="Europe/Warsaw",
                 max_json_length=500, trim_lists=True, verbose=False):
        """
        Inicjalizacja loggera.

        :param log_file: Ścieżka do pliku z logami. Jeśli None, logi nie będą zapisywane do pliku.
        :param console_level: Poziom logowania dla konsoli.
        :param file_level: Poziom logowania dla pliku.
        :param timezone: Strefa czasowa do formatowania czasu.
        :param max_json_length: Maksymalna długość logowanych JSONów przed ich przycięciem
        :param trim_lists: Czy przycinać długie listy w logach
        :param verbose: Czy logować szczegółowe informacje
        """
        self.timezone = pytz.timezone(timezone)
        self.console_level = console_level
        self.file_level = file_level
        self.log_file = log_file
        self.max_json_length = max_json_length
        self.trim_lists = trim_lists
        self.verbose = verbose

        # Konfiguracja loggera
        self.logger = logging.getLogger("StickFigureWebcam")
        self.logger.setLevel(5)  # Najniższy poziom (TRACE)
        self.logger.handlers = []  # Usuń wszystkie handlery

        # Dodaj handler konsoli
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.LEVELS[console_level]["level"])
        console_handler.setFormatter(self.ColoredFormatter())
        self.logger.addHandler(console_handler)

        # Dodaj handler pliku, jeśli podano
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(self.LEVELS[file_level]["level"])
            file_handler.setFormatter(self.PlainFormatter())
            self.logger.addHandler(file_handler)

        self.info("Logger", "Inicjalizacja loggera zakończona pomyślnie", log_type="CONFIG")

    def _format_message(self, level, module, message, log_type=None, plain=False):
        """Formatuje wiadomość logu."""
        now = datetime.datetime.now(self.timezone)
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        level_info = self.LEVELS[level]

        if plain:
            # Formatowanie bez kolorów dla plików logów
            formatted = f"[{time_str}] {level_info['symbol']} [{level}]"

            # Dodaj typ logu, jeśli podano
            if log_type and log_type in self.TYPES:
                type_info = self.TYPES[log_type]
                formatted += f" {type_info['symbol']} [{log_type}]"

            # Dodaj moduł i wiadomość
            formatted += f" [{module}] {message}"
        else:
            # Formatowanie z kolorami dla konsoli
            formatted = f"{level_info['color']}[{time_str}] {level_info['symbol']} [{level}]"

            # Dodaj typ logu, jeśli podano
            if log_type and log_type in self.TYPES:
                type_info = self.TYPES[log_type]
                formatted += f" {type_info['color']}{type_info['symbol']} [{log_type}]"

            # Dodaj moduł i wiadomość
            formatted += f" {Style.BRIGHT}{Fore.WHITE}[{module}]{Style.RESET_ALL} {message}"

        return formatted

    def _smart_trim(self, data, max_depth=2, current_depth=0):
        """
        Inteligentnie przycina złożone struktury danych, zachowując czytelność.
        """
        if current_depth >= max_depth:
            if isinstance(data, dict) and len(data) > 3:
                return {k: "..." for k, v in list(data.items())[:3]}
            elif isinstance(data, list) and len(data) > 3:
                return data[:3] + ["... (i {} więcej elementów)".format(len(data) - 3)]
            else:
                return data

        if isinstance(data, dict):
            return {k: self._smart_trim(v, max_depth, current_depth + 1) for k, v in data.items()}
        elif isinstance(data, list) and self.trim_lists and len(data) > 5:
            return [self._smart_trim(x, max_depth, current_depth + 1) for x in data[:5]] + \
                ["... (i {} więcej elementów)".format(len(data) - 5)]
        elif isinstance(data, list):
            return [self._smart_trim(x, max_depth, current_depth + 1) for x in data]
        else:
            return data

    def _format_frame_data(self, data):
        """
        Inteligentnie przetwarza dane klatki wideo, pozostawiając tylko najważniejsze informacje.
        """
        if not self.verbose:
            # Jeśli nie chcemy pełnych danych, wyciągamy kluczowe informacje
            important_data = {}

            # Zapisujemy najważniejsze pola
            for key in ["resolution", "fps", "frame_number"]:
                if key in data:
                    important_data[key] = data[key]

            # Dane o punktach charakterystycznych (landmarks)
            if "landmarks" in data and isinstance(data["landmarks"], list):
                if len(data["landmarks"]) > 5:
                    important_data["landmarks"] = f"{len(data['landmarks'])} punktów charakterystycznych"
                else:
                    important_data["landmarks"] = data["landmarks"]

            # Zawsze zachowujemy informacje o błędach
            if "error" in data:
                important_data["error"] = data["error"]

            return important_data
        else:
            # Jeśli chcemy pełne dane, inteligentnie przycinamy
            return self._smart_trim(data)

    def _log_json(self, data, max_length=None):
        """
        Inteligentne logowanie danych JSON z ograniczeniem długości.
        """
        if max_length is None:
            max_length = self.max_json_length

        try:
            json_text = json.dumps(data, indent=2, ensure_ascii=False)

            if len(json_text) > max_length:
                # Jeśli tekst jest za długi, pokazujemy początek i koniec
                half_length = max_length // 2 - 10
                return json_text[:half_length] + "\n...\n[skrócono " + str(
                    len(json_text) - max_length) + " znaków]\n..." + json_text[-half_length:]
            return json_text
        except Exception as e:
            return f"<błąd formatowania JSON: {e}>"

    def _log(self, level, module, message, log_type=None, **kwargs):
        """Zapisuje log z określonym poziomem."""
        # Tworzenie dwóch formatów wiadomości - z kolorami i bez kolorów
        formatted = self._format_message(level, module, message, log_type, plain=False)
        plain_formatted = self._format_message(level, module, message, log_type, plain=True)

        # Niestandardowa obsługa log recordu, aby przechować obie wersje wiadomości
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=self.LEVELS[level]["level"] if level != "TRACE" else 5,
            pathname="",
            lineno=0,
            msg=formatted,
            args=(),
            exc_info=None
        )
        # Dodajemy plain_msg jako atrybut, który zostanie użyty przez PlainFormatter
        log_record.plain_msg = plain_formatted

        # Przekazanie rekordu do wszystkich handlerów
        for handler in self.logger.handlers:
            if handler.level <= log_record.levelno:
                handler.handle(log_record)

        # Jeśli są dodatkowe dane, wypisz je ładnie
        if kwargs:
            filtered_kwargs = {}

            # Przetwarzanie specjalnych pól
            for key, value in kwargs.items():
                if key == "frame_data" and not self.verbose:
                    # Dla danych klatki stosujemy specjalne przetwarzanie
                    filtered_kwargs[key] = self._format_frame_data(value)
                elif isinstance(value, (dict, list)):
                    # Dla złożonych struktur stosujemy inteligentne przycinanie
                    filtered_kwargs[key] = self._smart_trim(value)
                else:
                    # Wartości proste pozostawiamy bez zmian
                    filtered_kwargs[key] = value

            # Logujemy przetworzone dane
            self._log_data(level, **filtered_kwargs)

    def _log_data(self, level, **kwargs):
        """Loguje dodatkowe dane jako JSON."""
        log_level = 5 if level == "TRACE" else self.LEVELS[level]["level"]

        for key, value in kwargs.items():
            if value is not None:
                try:
                    # Przygotuj tekst dla konsoli (kolorowy)
                    console_prefix = f"{Fore.CYAN}[DATA] {key}:"

                    # Przygotuj tekst dla pliku (bez kolorów)
                    file_prefix = f"[DATA] {key}:"

                    # Logowanie w zależności od typu danych
                    if isinstance(value, (dict, list)):
                        formatted_json = self._log_json(value)

                        # Tworzenie rekordów logów z różnymi formatami dla konsoli i pliku
                        console_record = logging.LogRecord(
                            name=self.logger.name,
                            level=log_level,
                            pathname="",
                            lineno=0,
                            msg=f"{console_prefix}\n{formatted_json}",
                            args=(),
                            exc_info=None
                        )
                        console_record.plain_msg = f"{file_prefix}\n{formatted_json}"
                    else:
                        # Dla prostych wartości
                        console_record = logging.LogRecord(
                            name=self.logger.name,
                            level=log_level,
                            pathname="",
                            lineno=0,
                            msg=f"{console_prefix} {value}",
                            args=(),
                            exc_info=None
                        )
                        console_record.plain_msg = f"{file_prefix} {value}"

                    # Przekazanie rekordów do handlerów
                    for handler in self.logger.handlers:
                        if handler.level <= log_level:
                            handler.handle(console_record)

                except Exception as e:
                    self.logger.error(f"Błąd podczas logowania danych: {e}")

    def trace(self, module, message, log_type=None, **kwargs):
        """Log najdrobniejszych szczegółów (poziom TRACE)."""
        self._log("TRACE", module, message, log_type, **kwargs)

    def debug(self, module, message, log_type=None, **kwargs):
        """Log debugowania."""
        self._log("DEBUG", module, message, log_type, **kwargs)

    def info(self, module, message, log_type=None, **kwargs):
        """Log informacyjny."""
        self._log("INFO", module, message, log_type, **kwargs)

    def warning(self, module, message, log_type=None, **kwargs):
        """Log ostrzeżenia."""
        self._log("WARNING", module, message, log_type, **kwargs)

    def error(self, module, message, log_type=None, **kwargs):
        """Log błędu."""
        self._log("ERROR", module, message, log_type, **kwargs)

    def critical(self, module, message, log_type=None, **kwargs):
        """Log krytyczny."""
        self._log("CRITICAL", module, message, log_type, **kwargs)

    def camera_status(self, status, camera_info=None):
        """
        Specjalny log dla statusu kamery.

        Args:
            status (bool): True jeśli kamera jest dostępna, False w przeciwnym przypadku
            camera_info (dict): Informacje o kamerze
        """
        if status:
            self.info(
                "CameraStatus",
                f"Kamera dostępna - {camera_info.get('name', 'Unknown')}",
                log_type="CAMERA",
                resolution=camera_info.get("resolution"),
                fps=camera_info.get("fps")
            )
        else:
            self.error(
                "CameraStatus",
                f"Kamera niedostępna lub problem z dostępem",
                log_type="CAMERA",
                error=camera_info.get("error", "Unknown error") if camera_info else "No camera info"
            )

    def pose_detection(self, detection_status, pose_data=None):
        """
        Log detekcji pozy.

        Args:
            detection_status (bool): True jeśli poza została wykryta, False w przeciwnym przypadku
            pose_data (dict): Dane o wykrytej pozie
        """
        if detection_status:
            self.info(
                "PoseDetection",
                f"Wykryto pozę - {'Siedząca' if pose_data.get('sitting', False) else 'Stojąca'}",
                log_type="POSE",
                landmarks_count=pose_data.get("landmarks_count") if pose_data else None,
                confidence=pose_data.get("confidence") if pose_data else None
            )
        else:
            self.warning(
                "PoseDetection",
                f"Nie wykryto pozy",
                log_type="POSE",
                error=pose_data.get("error") if pose_data else "No pose data"
            )

    def performance_metrics(self, fps, processing_time, module_name):
        """
        Log metryk wydajnościowych.

        Args:
            fps (float): Liczba klatek na sekundę
            processing_time (float): Czas przetwarzania w ms
            module_name (str): Nazwa modułu
        """
        level = "INFO"
        if processing_time > 50:  # Powyżej 50ms (mniej niż 20 FPS) - ostrzeżenie
            level = "WARNING"
        if processing_time > 100:  # Powyżej 100ms (mniej niż 10 FPS) - błąd
            level = "ERROR"

        self._log(
            level,
            module_name,
            f"Wydajność: {fps:.1f} FPS (czas przetwarzania: {processing_time:.2f}ms)",
            log_type="PERFORMANCE",
            fps=fps,
            processing_time=processing_time
        )

    def virtual_camera_status(self, status, camera_info=None):
        """
        Log statusu wirtualnej kamery.

        Args:
            status (bool): True jeśli wirtualna kamera działa, False w przeciwnym przypadku
            camera_info (dict): Informacje o wirtualnej kamerze
        """
        if status:
            self.info(
                "VirtualCam",
                f"Wirtualna kamera uruchomiona pomyślnie",
                log_type="VIRTUAL_CAM",
                resolution=camera_info.get("resolution") if camera_info else None,
                fps=camera_info.get("fps") if camera_info else None,
                backend=camera_info.get("backend") if camera_info else None
            )
        else:
            self.error(
                "VirtualCam",
                f"Problem z wirtualną kamerą",
                log_type="VIRTUAL_CAM",
                error=camera_info.get("error") if camera_info else "No camera info"
            )
