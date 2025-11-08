#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/lighting/adaptive_colors.py
"""
Moduł analizy oświetlenia i adaptacyjnego dostosowywania kolorów.
Dba o to, by kolory tła i stick figure dostosowywały się do warunków oświetleniowych
otoczenia, zachowując stały poziom kontrastu.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger


class AdaptiveLightingManager:
    """
    Klasa zarządzająca adaptacyjnym dostosowywaniem kolorów do warunków oświetleniowych.
    """

    def __init__(
        self,
        adaptation_speed: float = 0.02,  # Bardzo powolna adaptacja (0.0-1.0)
        smoothing_window: int = 30,  # Okno wygładzania (liczba klatek)
        min_brightness: int = 20,  # Minimalna jasność tła (0-255)
        max_brightness: int = 250,  # Maksymalna jasność tła (0-255)
        min_contrast: float = 0.4,  # Minimalny kontrast (0.0-1.0)
        sampling_interval: int = 5,  # Co ile klatek pobierać próbkę (oszczędność CPU)
        logger: Optional[CustomLogger] = None,
    ):
        """
        Inicjalizacja managera adaptacyjnego oświetlenia.

        Args:
            adaptation_speed (float): Szybkość adaptacji kolorów (0.0-1.0)
            smoothing_window (int): Liczba klatek używana do wygładzania zmian
            min_brightness (int): Minimalna jasność tła (0-255)
            max_brightness (int): Maksymalna jasność tła (0-255)
            min_contrast (float): Minimalny kontrast między tłem a konturami (0.0-1.0)
            sampling_interval (int): Co ile klatek analizować jasność (oszczędność CPU)
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.logger = logger or CustomLogger()

        # Parametry adaptacji
        self.adaptation_speed = max(0.001, min(0.1, adaptation_speed))  # Ograniczamy zakres
        self.smoothing_window = smoothing_window
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.sampling_interval = max(1, sampling_interval)

        # Stan wewnętrzny
        self.brightness_history: List[float] = []
        self.current_frame_index = 0
        self.current_bg_color = (255, 255, 255)  # Domyślnie białe tło (BGR)
        self.current_figure_color = (0, 0, 0)  # Domyślnie czarny kontur (BGR)
        self.target_bg_brightness = 255
        self.last_update_time = time.time()

        self.logger.info(
            "AdaptiveLighting",
            f"Inicjalizacja managera adaptacyjnego oświetlenia (szybkość={adaptation_speed}, "
            f"okno={smoothing_window}, min_brightness={min_brightness})",
            log_type="LIGHTING",
        )

    def analyze_frame(self, frame: np.ndarray) -> float:
        """
        Analizuje jasność klatki wideo.

        Args:
            frame (np.ndarray): Klatka wejściowa (BGR)

        Returns:
            float: Średnia jasność klatki (0.0-1.0)
        """
        # Inkrementacja licznika klatek
        self.current_frame_index += 1

        # Analizujemy tylko co X-tą klatkę dla oszczędności CPU
        if self.current_frame_index % self.sampling_interval != 0:
            # Zwracamy ostatnią znaną wartość, jeśli dostępna
            if self.brightness_history:
                return self.brightness_history[-1]
            return 0.5  # Wartość domyślna

        try:
            # Konwersja do skali szarości
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Obliczenie średniej jasności (normalizacja do 0.0-1.0)
            mean_brightness = np.mean(gray) / 255.0

            # Dodanie do historii
            self.brightness_history.append(mean_brightness)

            # Ograniczenie rozmiaru historii
            if len(self.brightness_history) > self.smoothing_window:
                self.brightness_history.pop(0)

            # Wygładzona wartość (średnia z historii)
            smoothed_brightness = np.mean(self.brightness_history)

            return smoothed_brightness

        except Exception as e:
            self.logger.error(
                "AdaptiveLighting", f"Błąd podczas analizy jasności: {str(e)}", log_type="LIGHTING"
            )
            # Zwracamy ostatnią znaną wartość, lub wartość domyślną
            if self.brightness_history:
                return self.brightness_history[-1]
            return 0.5

    def update_colors(
        self, frame_brightness: float
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Aktualizuje kolory tła i konturu na podstawie jasności otoczenia.
        Zachowuje stały kontrast pomiędzy tłem a konturem.

        Args:
            frame_brightness (float): Jasność otoczenia (0.0-1.0)

        Returns:
            Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
                - kolor tła (BGR)
                - kolor konturu (BGR)
        """
        # Obliczamy docelową jasność tła na podstawie jasności otoczenia
        # Odwracamy skalę - im jaśniejsze otoczenie, tym jaśniejsze tło
        target_bg_value = self.min_brightness + frame_brightness * (
            self.max_brightness - self.min_brightness
        )

        # Płynna adaptacja do docelowej wartości
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        adaptation_factor = min(1.0, elapsed_time * self.adaptation_speed * 10)

        # Aktualizacja docelowej jasności z uwzględnieniem współczynnika adaptacji
        self.target_bg_brightness = (
            self.target_bg_brightness * (1 - adaptation_factor)
            + target_bg_value * adaptation_factor
        )

        # Ograniczenie do zakresu
        self.target_bg_brightness = max(
            self.min_brightness, min(self.max_brightness, self.target_bg_brightness)
        )

        # Konwersja do koloru BGR
        bg_color_value = int(self.target_bg_brightness)
        bg_color = (bg_color_value, bg_color_value, bg_color_value)  # Odcień szarości (BGR)

        # Obliczamy kolor konturu - odwrotny do tła, ale z zachowaniem kontrastu
        # Im jaśniejsze tło, tym ciemniejszy kontur i odwrotnie
        contour_value = self._calculate_contrasting_value(bg_color_value)
        figure_color = (contour_value, contour_value, contour_value)  # Odcień szarości (BGR)

        # Aktualizujemy stan
        self.current_bg_color = bg_color
        self.current_figure_color = figure_color
        self.last_update_time = current_time

        return bg_color, figure_color

    def _calculate_contrasting_value(self, bg_value: int) -> int:
        """
        Oblicza wartość koloru konturu kontrastującą z tłem.

        Args:
            bg_value (int): Wartość jasności tła (0-255)

        Returns:
            int: Wartość jasności konturu (0-255)
        """
        # Punkt środkowy skali jasności
        mid_point = 127.5

        # Obliczamy względną jasność tła (-1.0 do 1.0 gdzie 0 to środek skali)
        relative_brightness = (bg_value - mid_point) / mid_point

        # Odwracamy i skalujemy, zachowując minimalny kontrast
        contrast_factor = max(self.min_contrast, abs(relative_brightness) + self.min_contrast)

        if bg_value > mid_point:
            # Dla jasnego tła - ciemny kontur
            contour_value = max(0, int(bg_value * (1.0 - contrast_factor)))
        else:
            # Dla ciemnego tła - jasny kontur
            contour_value = min(255, int(bg_value * (1.0 + contrast_factor)))

        return contour_value

    def get_current_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Zwraca aktualnie używane kolory.

        Returns:
            Dict[str, Tuple[int, int, int]]: Słownik z kolorami
        """
        return {
            "bg_color": self.current_bg_color,
            "figure_color": self.current_figure_color,
            "brightness_level": self.target_bg_brightness / 255.0,
        }

    def reset(self) -> None:
        """
        Resetuje stan managera.
        """
        self.brightness_history = []
        self.current_frame_index = 0
        self.current_bg_color = (255, 255, 255)
        self.current_figure_color = (0, 0, 0)
        self.target_bg_brightness = 255
        self.last_update_time = time.time()
