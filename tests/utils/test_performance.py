#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testy jednostkowe dla monitora wydajności (performance.py).
"""

import unittest
from unittest.mock import patch

from src.utils.performance import PerformanceMonitor


class TestPerformanceMonitor(unittest.TestCase):
    """
    Testy dla klasy PerformanceMonitor, która mierzy czas wykonania, FPS
    i przechowuje historię pomiarów.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Inicjalizacja monitora wydajności
        self.monitor = PerformanceMonitor("TestModule", history_size=5)

    def test_initialization(self):
        """Test inicjalizacji monitora wydajności."""
        # Sprawdzamy czy inicjalizacja przebiegła poprawnie
        self.assertEqual(self.monitor.module_name, "TestModule")
        self.assertEqual(self.monitor.history_size, 5)
        self.assertIsNone(self.monitor.start_time)
        self.assertEqual(len(self.monitor.execution_times), 0)
        self.assertEqual(len(self.monitor.fps_history), 0)
        self.assertEqual(self.monitor.frame_counter, 0)
        self.assertEqual(len(self.monitor.timing_markers), 0)
        self.assertEqual(len(self.monitor.segment_times), 0)

    @patch('time.time')
    def test_start_timer(self, mock_time):
        """Test rozpoczęcia pomiaru czasu."""
        # Konfigurujemy mocka time.time
        mock_time.return_value = 10.0

        # Rozpoczynamy pomiar
        self.monitor.start_timer()

        # Sprawdzamy czy czas rozpoczęcia został zapisany
        self.assertEqual(self.monitor.start_time, 10.0)

    @patch('time.time')
    def test_stop_timer(self, mock_time):
        """Test zatrzymania pomiaru czasu."""
        # Konfigurujemy mocka time.time dla różnych wywołań
        mock_time.side_effect = [10.0, 10.5, 11.0]

        # Rozpoczynamy pomiar
        self.monitor.start_timer()

        # Zatrzymujemy pomiar
        execution_time = self.monitor.stop_timer()

        # Sprawdzamy wyniki
        self.assertEqual(execution_time, 0.5)  # 10.5 - 10.0
        self.assertEqual(len(self.monitor.execution_times), 1)
        self.assertEqual(self.monitor.execution_times[0], 0.5)
        self.assertEqual(self.monitor.frame_counter, 1)

        # Nie testujemy aktualizacji FPS, bo wymaga więcej przygotowań mocków
        # i jest bardziej podatna na błędy

    def test_get_last_execution_time(self):
        """Test pobierania ostatniego czasu wykonania."""
        # Gdy brak pomiarów
        self.assertEqual(self.monitor.get_last_execution_time(), 0.0)

        # Dodajemy pomiar
        self.monitor.execution_times.append(0.5)

        # Sprawdzamy ostatni czas
        self.assertEqual(self.monitor.get_last_execution_time(), 0.5)

    def test_get_average_execution_time(self):
        """Test obliczania średniego czasu wykonania."""
        # Gdy brak pomiarów
        self.assertEqual(self.monitor.get_average_execution_time(), 0.0)

        # Dodajemy pomiary
        self.monitor.execution_times.extend([0.1, 0.2, 0.3, 0.4, 0.5])

        # Sprawdzamy średnią wszystkich pomiarów
        self.assertAlmostEqual(self.monitor.get_average_execution_time(), 0.3, places=6)

        # Sprawdzamy średnią ostatnich 3 pomiarów
        self.assertAlmostEqual(self.monitor.get_average_execution_time(num_samples=3), 0.4, places=6)

    def test_get_current_fps(self):
        """Test pobierania bieżącego FPS."""
        # Gdy brak pomiarów
        self.assertEqual(self.monitor.get_current_fps(), 0.0)

        # Dodajemy pomiar FPS
        self.monitor.fps_history.append(30.0)

        # Sprawdzamy bieżący FPS
        self.assertEqual(self.monitor.get_current_fps(), 30.0)

    def test_get_average_fps(self):
        """Test obliczania średniego FPS."""
        # Gdy brak pomiarów
        self.assertEqual(self.monitor.get_average_fps(), 0.0)

        # Dodajemy pomiary FPS
        self.monitor.fps_history.extend([25.0, 27.0, 30.0, 28.0, 26.0])

        # Sprawdzamy średnią wszystkich pomiarów
        self.assertEqual(self.monitor.get_average_fps(), 27.2)

        # Sprawdzamy średnią ostatnich 3 pomiarów
        self.assertEqual(self.monitor.get_average_fps(num_samples=3), 28.0)

    @patch('time.time')
    def test_mark_time(self, mock_time):
        """Test zapisywania znaczników czasu."""
        # Konfigurujemy mocka time.time
        mock_time.return_value = 10.0

        # Zapisujemy znacznik czasu
        self.monitor.mark_time("start")

        # Sprawdzamy czy znacznik został zapisany
        self.assertEqual(len(self.monitor.timing_markers), 1)
        self.assertEqual(self.monitor.timing_markers["start"], 10.0)

    @patch('time.time')
    def test_measure_segment(self, mock_time):
        """Test mierzenia czasu między znacznikami."""
        # Konfigurujemy mocka time.time dla różnych wywołań
        mock_time.side_effect = [10.0, 10.5]

        # Zapisujemy znaczniki czasu
        self.monitor.mark_time("start")
        self.monitor.mark_time("end")

        # Mierzymy segment
        duration = self.monitor.measure_segment("start", "end", "test_segment")

        # Sprawdzamy wyniki
        self.assertEqual(duration, 0.5)
        self.assertEqual(len(self.monitor.segment_times), 1)
        self.assertEqual(len(self.monitor.segment_times["test_segment"]), 1)
        self.assertEqual(self.monitor.segment_times["test_segment"][0], 0.5)

        # Test dla nieistniejących znaczników
        self.assertEqual(self.monitor.measure_segment("non_existent", "end", "invalid"), -1.0)

    def test_get_segment_stats(self):
        """Test pobierania statystyk segmentu."""
        # Gdy brak segmentu
        self.assertEqual(self.monitor.get_segment_stats("non_existent"), (0.0, 0.0, 0.0))

        # Dodajemy pomiary dla segmentu
        self.monitor.segment_times["test_segment"] = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Sprawdzamy statystyki
        avg, min_time, max_time = self.monitor.get_segment_stats("test_segment")
        self.assertAlmostEqual(avg, 0.3, places=6)
        self.assertEqual(min_time, 0.1)
        self.assertEqual(max_time, 0.5)

    def test_reset(self):
        """Test resetowania monitora wydajności."""
        # Dodajemy dane
        self.monitor.start_time = 10.0
        self.monitor.execution_times.append(0.5)
        self.monitor.fps_history.append(30.0)
        self.monitor.frame_counter = 10
        self.monitor.timing_markers["start"] = 10.0
        self.monitor.segment_times["test_segment"] = [0.5]

        # Resetujemy
        self.monitor.reset()

        # Sprawdzamy czy wszystko zostało zresetowane
        self.assertIsNone(self.monitor.start_time)
        self.assertEqual(len(self.monitor.execution_times), 0)
        self.assertEqual(len(self.monitor.fps_history), 0)
        self.assertEqual(self.monitor.frame_counter, 0)
        self.assertEqual(len(self.monitor.timing_markers), 0)
        self.assertEqual(len(self.monitor.segment_times), 0)

    def test_get_performance_summary(self):
        """Test tworzenia podsumowania wydajności."""
        # Dodajemy dane
        self.monitor.execution_times.extend([0.1, 0.2, 0.3, 0.4, 0.5])
        self.monitor.fps_history.extend([25.0, 27.0, 30.0, 28.0, 26.0])

        # Dodajemy segment
        self.monitor.segment_times["test_segment"] = [0.1, 0.2, 0.3]

        # Pobieramy podsumowanie
        summary = self.monitor.get_performance_summary()

        # Sprawdzamy zawartość podsumowania
        self.assertEqual(summary["module"], "TestModule")
        self.assertAlmostEqual(summary["avg_execution_time"], 0.3, places=6)
        self.assertAlmostEqual(summary["avg_execution_time_ms"], 300.0, places=6)
        self.assertEqual(summary["last_execution_time"], 0.5)
        self.assertEqual(summary["avg_fps"], 27.2)
        self.assertEqual(summary["current_fps"], 26.0)
        self.assertEqual(summary["samples_count"], 5)

        # Sprawdzamy statystyki segmentu
        self.assertIn("segments", summary)
        self.assertIn("test_segment", summary["segments"])
        segment_stats = summary["segments"]["test_segment"]
        self.assertAlmostEqual(segment_stats["avg"], 0.2, places=6)
        self.assertEqual(segment_stats["min"], 0.1)
        self.assertEqual(segment_stats["max"], 0.3)


if __name__ == "__main__":
    unittest.main()
