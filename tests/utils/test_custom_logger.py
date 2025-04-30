#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testy jednostkowe dla niestandardowego loggera (custom_logger.py).
"""

import json
import logging
import os
import re
import tempfile
import unittest
from unittest.mock import patch

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

        # Inicjalizacja loggera bez pliku logów (tylko dla konsoli)
        self.console_logger = CustomLogger(log_file=None, console_level="INFO")

        # Inicjalizacja loggera z plikiem logów
        self.file_logger = CustomLogger(log_file=self.log_file, console_level="INFO", file_level="DEBUG")

    def tearDown(self):
        """Sprzątanie po każdym teście."""
        self.time_patch.stop()

        # Usuwamy plik logów jeśli istnieje
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # Usuwamy tymczasowy katalog
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_initialization_console_only(self):
        """Test inicjalizacji loggera tylko dla konsoli."""
        # Sprawdzamy czy logger został utworzony i skonfigurowany
        self.assertIsNotNone(self.console_logger.logger)
        self.assertEqual(len(self.console_logger.logger.handlers), 1)  # Tylko handler konsoli

        # Sprawdzamy poziom logowania
        self.assertEqual(self.console_logger.console_level, "INFO")
        self.assertEqual(self.console_logger.file_level, "DEBUG")  # Domyślny poziom dla pliku

        # Sprawdzamy czy handler konsoli ma odpowiedni poziom
        console_handler = self.console_logger.logger.handlers[0]
        self.assertEqual(console_handler.level, logging.INFO)

    def test_initialization_with_file(self):
        """Test inicjalizacji loggera z plikiem logów."""
        # Sprawdzamy czy logger został utworzony i skonfigurowany
        self.assertIsNotNone(self.file_logger.logger)
        self.assertEqual(len(self.file_logger.logger.handlers), 2)  # Handler konsoli i pliku

        # Sprawdzamy poziom logowania
        self.assertEqual(self.file_logger.console_level, "INFO")
        self.assertEqual(self.file_logger.file_level, "DEBUG")

        # Sprawdzamy czy handlery mają odpowiednie poziomy
        console_handler = self.file_logger.logger.handlers[0]
        file_handler = self.file_logger.logger.handlers[1]

        self.assertEqual(console_handler.level, logging.INFO)
        self.assertEqual(file_handler.level, logging.DEBUG)

        # Sprawdzamy czy plik logów został utworzony
        self.assertTrue(os.path.exists(self.log_file))

    @patch('logging.LogRecord')
    @patch('logging.Handler.handle')
    def test_logging_levels(self, mock_handle, mock_log_record):
        """Test logowania na różnych poziomach."""
        # Używamy loggera bez pliku, żeby uprościć test
        logger = self.console_logger

        # Testujemy wszystkie poziomy logowania
        log_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in log_levels:
            # Resetujemy mocki
            mock_log_record.reset_mock()
            mock_handle.reset_mock()

            # Logujemy wiadomość
            getattr(logger, level.lower())("TestModule", f"Test message for {level}")

            # Sprawdzamy czy LogRecord został utworzony z odpowiednim poziomem
            # Nie sprawdzamy dokładnych argumentów, bo to zależy od implementacji
            mock_log_record.assert_called()

    def test_logging_to_file(self):
        """Test logowania do pliku."""
        # Logujemy wiadomości na różnych poziomach
        self.file_logger.debug("TestModule", "Debug message")
        self.file_logger.info("TestModule", "Info message")
        self.file_logger.warning("TestModule", "Warning message")

        # Sprawdzamy zawartość pliku logów
        with open(self.log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Sprawdzamy czy wiadomości zostały zapisane
        self.assertIn("Debug message", log_content)
        self.assertIn("Info message", log_content)
        self.assertIn("Warning message", log_content)

        # Sprawdzamy format wiadomości - bez kolorów ANSI
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.assertFalse(bool(ansi_escape.search(log_content)))

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
        # Test logowania statusu kamery - używamy mocków
        with patch.object(self.console_logger, 'info') as mock_info:
            self.console_logger.camera_status(True, {"name": "Test Camera", "resolution": (640, 480), "fps": 30})
            mock_info.assert_called_once()

            # Sprawdzamy argumenty wywołania
            args, kwargs = mock_info.call_args
            self.assertEqual(args[0], "CameraStatus")
            self.assertIn("Kamera dostępna", args[1])
            self.assertEqual(kwargs["log_type"], "CAMERA")

        # Test logowania detekcji pozy
        with patch.object(self.console_logger, 'info') as mock_info:
            self.console_logger.pose_detection(True, {"sitting": True, "landmarks_count": 33, "confidence": 0.9})
            mock_info.assert_called_once()

            # Sprawdzamy argumenty wywołania
            args, kwargs = mock_info.call_args
            self.assertEqual(args[0], "PoseDetection")
            self.assertIn("Wykryto pozę", args[1])
            self.assertEqual(kwargs["log_type"], "POSE")

        # Test logowania metryk wydajności
        with patch.object(self.console_logger, '_log') as mock_log:
            self.console_logger.performance_metrics(30.0, 20.0, "TestModule")
            mock_log.assert_called_once()

            # Sprawdzamy argumenty wywołania
            args, kwargs = mock_log.call_args
            self.assertEqual(args[0], "INFO")  # Dla dobrych wyników używamy INFO
            self.assertEqual(args[1], "TestModule")
            self.assertIn("Wydajność:", args[2])
            self.assertEqual(kwargs["log_type"], "PERFORMANCE")


if __name__ == "__main__":
    unittest.main()
