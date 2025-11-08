#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from collections import deque
from typing import Dict, Optional, Tuple


class PerformanceMonitor:
    """
    Narzędzie do monitorowania wydajności różnych operacji w aplikacji.
    Mierzy czas wykonania, FPS i przechowuje historię pomiarów.
    """

    def __init__(self, module_name: str, history_size: int = 100):
        """
        Inicjalizacja monitora wydajności.

        Args:
            module_name (str): Nazwa monitorowanego modułu
            history_size (int): Liczba przechowywanych historycznych pomiarów
        """
        self.module_name = module_name
        self.history_size = history_size

        # Dane o pomiarach
        self.start_time: Optional[float] = None
        self.execution_times = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        self.last_fps_update_time = time.time()
        self.frame_counter = 0

        # Znaczniki czasu dla przechowywania wielu różnych pomiarów
        self.timing_markers: Dict[str, float] = {}
        self.segment_times: Dict[str, deque] = {}

    def start_timer(self) -> None:
        """
        Rozpoczyna pomiar czasu wykonania.
        """
        self.start_time = time.time()

    def stop_timer(self) -> float:
        """
        Zatrzymuje pomiar czasu i zapisuje czas wykonania w historii.

        Returns:
            float: Czas wykonania w sekundach
        """
        if self.start_time is None:
            return 0.0

        execution_time = time.time() - self.start_time
        self.execution_times.append(execution_time)

        # Aktualizacja licznika klatek
        self.frame_counter += 1

        # Obliczanie FPS co sekundę
        current_time = time.time()
        time_diff = current_time - self.last_fps_update_time

        if time_diff >= 1.0:
            fps = self.frame_counter / time_diff
            self.fps_history.append(fps)
            self.last_fps_update_time = current_time
            self.frame_counter = 0

        return execution_time

    def get_last_execution_time(self) -> float:
        """
        Zwraca ostatni zmierzony czas wykonania.

        Returns:
            float: Czas wykonania w sekundach lub 0.0 jeśli brak pomiarów
        """
        if not self.execution_times:
            return 0.0
        return self.execution_times[-1]

    def get_average_execution_time(self, num_samples: Optional[int] = None) -> float:
        """
        Oblicza średni czas wykonania.

        Args:
            num_samples (Optional[int]): Liczba ostatnich próbek do uwzględnienia
                                        Jeśli None, używa wszystkich dostępnych

        Returns:
            float: Średni czas wykonania w sekundach
        """
        if not self.execution_times:
            return 0.0

        if num_samples is None or num_samples >= len(self.execution_times):
            return sum(self.execution_times) / len(self.execution_times)
        else:
            samples = list(self.execution_times)[-num_samples:]
            return sum(samples) / len(samples)

    def get_current_fps(self) -> float:
        """
        Zwraca ostatnią obliczoną wartość FPS.

        Returns:
            float: Liczba klatek na sekundę lub 0.0 jeśli brak danych
        """
        if not self.fps_history:
            return 0.0
        return self.fps_history[-1]

    def get_average_fps(self, num_samples: Optional[int] = None) -> float:
        """
        Oblicza średni FPS.

        Args:
            num_samples (Optional[int]): Liczba ostatnich próbek do uwzględnienia
                                        Jeśli None, używa wszystkich dostępnych

        Returns:
            float: Średni FPS
        """
        if not self.fps_history:
            return 0.0

        if num_samples is None or num_samples >= len(self.fps_history):
            return sum(self.fps_history) / len(self.fps_history)
        else:
            samples = list(self.fps_history)[-num_samples:]
            return sum(samples) / len(samples)

    def mark_time(self, marker_name: str) -> None:
        """
        Zapisuje znacznik czasu o określonej nazwie.
        Przydatne do mierzenia czasu wykonania poszczególnych segmentów kodu.

        Args:
            marker_name (str): Nazwa znacznika
        """
        self.timing_markers[marker_name] = time.time()

    def measure_segment(self, start_marker: str, end_marker: str, segment_name: str) -> float:
        """
        Mierzy czas między dwoma znacznikami i zapisuje jako segment.

        Args:
            start_marker (str): Nazwa znacznika początkowego
            end_marker (str): Nazwa znacznika końcowego
            segment_name (str): Nazwa segmentu do zapisania

        Returns:
            float: Czas trwania segmentu w sekundach lub -1.0 jeśli brak znaczników
        """
        if start_marker not in self.timing_markers or end_marker not in self.timing_markers:
            return -1.0

        duration = self.timing_markers[end_marker] - self.timing_markers[start_marker]

        if segment_name not in self.segment_times:
            self.segment_times[segment_name] = deque(maxlen=self.history_size)

        self.segment_times[segment_name].append(duration)
        return duration

    def get_segment_stats(self, segment_name: str) -> Tuple[float, float, float]:
        """
        Zwraca statystyki czasów wykonania dla danego segmentu.

        Args:
            segment_name (str): Nazwa segmentu

        Returns:
            Tuple[float, float, float]: Średni, minimalny i maksymalny czas w sekundach
                                        Zwraca (0.0, 0.0, 0.0) jeśli brak danych
        """
        if segment_name not in self.segment_times or not self.segment_times[segment_name]:
            return 0.0, 0.0, 0.0

        times = self.segment_times[segment_name]
        return (sum(times) / len(times), min(times), max(times))  # średnia  # minimum  # maksimum

    def reset(self) -> None:
        """
        Resetuje wszystkie pomiary.
        """
        self.start_time = None
        self.execution_times.clear()
        self.fps_history.clear()
        self.last_fps_update_time = time.time()
        self.frame_counter = 0
        self.timing_markers.clear()
        self.segment_times.clear()

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Tworzy podsumowanie wydajności.

        Returns:
            Dict[str, float]: Słownik z podsumowaniem wydajności
        """
        avg_time = self.get_average_execution_time()
        avg_fps = self.get_average_fps()

        summary = {
            "module": self.module_name,
            "avg_execution_time": avg_time,
            "avg_execution_time_ms": avg_time * 1000,
            "last_execution_time": self.get_last_execution_time(),
            "last_execution_time_ms": self.get_last_execution_time() * 1000,
            "avg_fps": avg_fps,
            "current_fps": self.get_current_fps(),
            "samples_count": len(self.execution_times),
        }

        # Dodaj statystyki segmentów jeśli istnieją
        segments = {}
        for segment_name in self.segment_times:
            avg, min_time, max_time = self.get_segment_stats(segment_name)
            segments[segment_name] = {
                "avg": avg,
                "avg_ms": avg * 1000,
                "min": min_time,
                "min_ms": min_time * 1000,
                "max": max_time,
                "max_ms": max_time * 1000,
            }

        if segments:
            summary["segments"] = segments

        return summary
