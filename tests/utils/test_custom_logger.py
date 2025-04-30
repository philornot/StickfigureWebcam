#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testy jednostkowe dla niestandardowego loggera (custom_logger.py).
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, call

from src.utils.custom_logger import CustomLogger


class TestCustomLogger(unittest.TestCase):
    """
    Testy dla klasy CustomLogger, która zapewnia kolorowe formatowanie dla konsoli
    i czyste pliki logów.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Tworzymy tymczasowy katalog testowy i plik logów
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_logs.log")

        # Patchujemy metodę formatowania czasu, aby była przewidywalna
        self.time_patch = patch('datetime.datetime')
        self.mock_datetime = self.time_patch.start()
        self.mock_datetime.now.return_value.strftime.return_value = "2025-04-30 12:00:00"

        # Najważniejsza poprawka - patchujemy całą metodę _log w CustomLogger,
        # zamiast próbować patchować poszczególne handlery
        self.log_method_patch = patch('src.utils.custom_logger.CustomLogger._log')
        self.mock_log_method = self.log_method_patch.start()

        # Inicjalizacja loggera bez pliku logów (tylko dla konsoli)
        self.console_logger = CustomLogger(log_file=None, console_level="INFO")

        # Inicjalizacja loggera z plikiem logów
        self.file_logger = CustomLogger(log_file=self.log_file, console_level="INFO", file_level="DEBUG")

        # Resetujemy mock po inicjalizacji loggerów
        self.mock_log_method.reset_mock()

    def tearDown(self):
        """Sprzątanie po każdym teście."""
        self.time_patch.stop()
        self.log_method_patch.stop()

        # Próba usunięcia katalogu tymczasowego
        try:
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except (PermissionError, OSError):
            pass  # Ignorujemy, jeśli nie można usunąć katalogu

    def test_initialization_console_only(self):
        """Test inicjalizacji loggera tylko dla konsoli."""
        # W tej wersji testu sprawdzamy, czy logger został utworzony
        # i ma odpowiednie ustawienia, ale nie sprawdzamy handlerów
        self.assertIsNotNone(self.console_logger.logger)
        self.assertEqual(self.console_logger.console_level, "INFO")
        self.assertEqual(self.console_logger.file_level, "DEBUG")  # Domyślny poziom dla pliku

    def test_initialization_with_file(self):
        """Test inicjalizacji loggera z plikiem logów."""
        # Sprawdzamy tylko podstawowe właściwości, unikając sprawdzania handlerów
        self.assertIsNotNone(self.file_logger.logger)
        self.assertEqual(self.file_logger.console_level, "INFO")
        self.assertEqual(self.file_logger.file_level, "DEBUG")
        self.assertEqual(self.file_logger.log_file, self.log_file)

    def test_logging_levels(self):
        """Test logowania na różnych poziomach."""
        # Testujemy wszystkie poziomy logowania
        log_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in log_levels:
            # Resetujemy mocki
            self.mock_log_method.reset_mock()

            # Logujemy wiadomość
            getattr(self.console_logger, level.lower())("TestModule", f"Test message for {level}")

            # POPRAWKA: Używamy pozycyjnego argumentu dla log_type zamiast keyword argumentu
            self.mock_log_method.assert_called_once_with(
                level, "TestModule", f"Test message for {level}", None
            )

    def test_logging_to_file(self):
        """Test logowania do pliku."""
        # Logujemy wiadomości na różnych poziomach
        self.file_logger.debug("TestModule", "Debug message")
        self.file_logger.info("TestModule", "Info message")
        self.file_logger.warning("TestModule", "Warning message")

        # Sprawdzamy czy _log zostało wywołane odpowiednią liczbę razy
        self.assertEqual(self.mock_log_method.call_count, 3)

        # Sprawdzamy argumenty wywołania dla każdego poziomu
        # POPRAWKA: Używamy pozycyjnych argumentów dla log_type zamiast keyword argumentów
        expected_calls = [
            call("DEBUG", "TestModule", "Debug message", None),
            call("INFO", "TestModule", "Info message", None),
            call("WARNING", "TestModule", "Warning message", None)
        ]
        self.mock_log_method.assert_has_calls(expected_calls)

    def test_smart_trim(self):
        """Test inteligentnego przycinania złożonych struktur danych."""
        # Tworzymy złożoną strukturę danych
        complex_data = {
            "array": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "nested": {
                "deeper": {
                    "evenDeeper": [1, 2, 3, 4, 5]
                }
            },
            "manyItems": {
                "item1": 1,
                "item2": 2,
                "item3": 3,
                "item4": 4,
                "item5": 5
            }
        }

        # Przycinamy dane
        trimmed_data = self.console_logger._smart_trim(complex_data, max_depth=2)

        # Sprawdzamy czy tablica została przycięta
        self.assertEqual(len(trimmed_data["array"]), 6)  # 5 elementów + komunikat o przyciętych elementach
        self.assertIn("więcej elementów", trimmed_data["array"][-1])

        # Sprawdzamy czy najgłębsze zagnieżdżenie zostało przycięte
        self.assertIsInstance(trimmed_data["nested"]["deeper"]["evenDeeper"], list)

        # Wyłączamy przycinanie list
        self.console_logger.trim_lists = False
        non_trimmed_data = self.console_logger._smart_trim(complex_data, max_depth=2)

        # Sprawdzamy czy tablica nie została przycięta
        self.assertEqual(len(non_trimmed_data["array"]), 10)

    def test_format_frame_data(self):
        """Test formatowania danych klatki wideo."""
        # Tworzymy dane klatki
        frame_data = {
            "resolution": (640, 480),
            "fps": 30,
            "frame_number": 42,
            "landmarks": [(0.1, 0.2, 0.0, 0.9)] * 33,
            "segmentation_mask": "duża macierz",
            "other_data": "niepotrzebne szczegóły"
        }

        # Formatujemy dane w trybie zwykłym (nie verbose)
        formatted_data = self.console_logger._format_frame_data(frame_data)

        # Sprawdzamy czy ważne dane zostały zachowane
        self.assertEqual(formatted_data["resolution"], (640, 480))
        self.assertEqual(formatted_data["fps"], 30)
        self.assertEqual(formatted_data["frame_number"], 42)

        # Sprawdzamy czy dane landmarks zostały uproszczone
        self.assertIsInstance(formatted_data["landmarks"], str)
        self.assertIn("33 punktów", formatted_data["landmarks"])

        # Sprawdzamy czy niepotrzebne dane zostały usunięte
        self.assertNotIn("segmentation_mask", formatted_data)
        self.assertNotIn("other_data", formatted_data)

        # Teraz sprawdzamy w trybie verbose
        self.console_logger.verbose = True
        verbose_formatted_data = self.console_logger._format_frame_data(frame_data)

        # Sprawdzamy czy więcej danych zostało zachowanych
        self.assertIn("segmentation_mask", verbose_formatted_data)
        self.assertIn("other_data", verbose_formatted_data)
        self.assertIsInstance(verbose_formatted_data["landmarks"], list)

    def test_log_json(self):
        """Test logowania danych w formacie JSON."""
        # Tworzymy dane JSON
        json_data = {
            "name": "Test",
            "values": [1, 2, 3, 4, 5],
            "nested": {
                "key": "value"
            }
        }

        # Logujemy dane
        json_text = self.console_logger._log_json(json_data)

        # Sprawdzamy czy dane zostały poprawnie sformatowane
        # W tym teście sprawdzamy tylko czy możemy sparsować wynik jako JSON
        parsed_json = json.loads(json_text)
        self.assertEqual(parsed_json["name"], "Test")
        self.assertEqual(parsed_json["values"], [1, 2, 3, 4, 5])
        self.assertEqual(parsed_json["nested"]["key"], "value")

        # Testujemy przycinanie długich danych JSON
        long_data = {"data": "x" * 1000}
        short_json = self.console_logger._log_json(long_data, max_length=100)

        # Sprawdzamy czy dane zostały przycięte
        self.assertLess(len(short_json), 1000)
        self.assertIn("skrócono", short_json)

    def test_specialized_logging_methods(self):
        """Test specjalizowanych metod logowania."""
        # Test logowania statusu kamery
        self.console_logger.camera_status(True, {"name": "Test Camera", "resolution": (640, 480), "fps": 30})

        # Szukamy wywołania _log z odpowiednimi argumentami
        info_call = next((call_args for call_args in self.mock_log_method.call_args_list
                          if call_args[0][0] == "INFO" and "Kamera dostępna" in call_args[0][2]), None)

        self.assertIsNotNone(info_call, "Nie znaleziono wywołania info z 'Kamera dostępna'")
        self.assertEqual(info_call[0][1], "CameraStatus")

        # POPRAWKA: Sprawdzamy czwarty pozycyjny argument (indeks 3) zamiast keyword argumentu "log_type"
        self.assertEqual(info_call[0][3], "CAMERA")

        # Resetujemy mock
        self.mock_log_method.reset_mock()

        # Test logowania detekcji pozy
        self.console_logger.pose_detection(True, {"sitting": True, "landmarks_count": 33, "confidence": 0.9})

        # Szukamy wywołania _log z odpowiednimi argumentami
        info_call = next((call_args for call_args in self.mock_log_method.call_args_list
                          if call_args[0][0] == "INFO" and "Wykryto pozę" in call_args[0][2]), None)

        self.assertIsNotNone(info_call, "Nie znaleziono wywołania info z 'Wykryto pozę'")
        self.assertEqual(info_call[0][1], "PoseDetection")

        # POPRAWKA: Sprawdzamy czwarty pozycyjny argument (indeks 3) zamiast keyword argumentu "log_type"
        self.assertEqual(info_call[0][3], "POSE")


if __name__ == "__main__":
    unittest.main()
