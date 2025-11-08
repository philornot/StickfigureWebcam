#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/face_renderer.py

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger


class SimpleFaceRenderer:
    """
    Uproszczony renderer twarzy dla stick figure z podstawowymi wyrazami mimicznymi.
    Zapewnia płynne przejścia między wyrazami twarzy i zwiększoną czułość na zmiany mimiki.
    """

    def __init__(
        self,
        feature_color: Tuple[int, int, int] = (0, 0, 0),  # Czarny
        smooth_factor: float = 0.3,
        logger: Optional[CustomLogger] = None,
    ):
        """
        Inicjalizacja renderera twarzy.

        Args:
            feature_color (Tuple[int, int, int]): Kolor elementów twarzy (BGR)
            smooth_factor (float): Współczynnik wygładzania ruchu (0.0-1.0)
            logger (Optional[CustomLogger]): Logger do zapisywania komunikatów
        """
        # Inicjalizacja loggera
        self.logger = logger or CustomLogger()

        # Parametry renderowania
        self.feature_color = feature_color
        self.smooth_factor = smooth_factor

        # Bufor do wygładzania wyrazów twarzy
        self.last_expressions = {
            "mouth_open": 0.0,
            "smile": 0.5,  # 0.5 to neutralny uśmiech
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0,
        }

        # Dodanie histerezy, aby uniknąć przełączania między uśmiechem i smutkiem
        self.smile_histeresis = 0.05  # Zmniejszony margines histerezy dla większej czułości

        # Licznik klatek do logowania
        self.frame_count = 0

        # Zakres neutralnej miny (zmniejszony dla większej czułości)
        self.neutral_lower = 0.45
        self.neutral_upper = 0.55

        # Dodajemy filtr uśredniający ostatnie N wyrazów twarzy dla większej płynności
        self.expressions_history: List[Dict[str, float]] = []
        self.expressions_history_size = 5  # Ilość klatek do uśredniania

        self.logger.info(
            "SimpleFaceRenderer", "Zainicjalizowano renderer twarzy", log_type="DRAWING"
        )

    def draw_face(
        self,
        canvas: np.ndarray,
        head_center: Tuple[int, int],
        head_radius: int,
        mood: str = "neutral",
        face_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Rysuje twarz na stick figure.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            head_center (Tuple[int, int]): Współrzędne środka głowy (x, y)
            head_radius (int): Promień głowy
            mood (str): Nastrój do wyrażenia ("happy", "sad", "neutral", "surprised", "wink")
            face_data (Optional[Dict[str, Any]]): Dane twarzy z detektora twarzy (jeśli dostępne)
        """
        self.frame_count += 1

        try:
            # Środek głowy
            cx, cy = head_center

            # Jeśli mamy dane twarzy, używamy ich do animacji
            if face_data and "expressions" in face_data:
                expressions = face_data["expressions"]

                # Sprawdzamy czy wartości są sensowne
                for key, value in expressions.items():
                    if not (0.0 <= value <= 1.0):
                        # Jeśli wartość jest poza zakresem 0-1, używamy ostatniej poprawnej wartości
                        expressions[key] = self.last_expressions.get(key, 0.5)

                # Dodajemy aktualne wyrazy do historii
                self.expressions_history.append(expressions)
                if len(self.expressions_history) > self.expressions_history_size:
                    self.expressions_history.pop(0)

                # Uśredniamy wyrazy z historii dla większej płynności
                avg_expressions = self._average_expressions()

                # Następnie wygładzamy wyrazy twarzy
                smoothed_expressions = self._smooth_expressions(avg_expressions)

                # Następnie stosujemy histerezę dla uśmiechu, aby uniknąć przełączania
                smile_value = smoothed_expressions["smile"]

                # Zaktualizuj wartość nastroju opartą na uśmiechu z histerezą
                current_mood = self._determine_mood_from_smile(smile_value)

                # Jeśli nie ma wymuszonego nastroju (gdy mood nie jest jednym z predefiniowanych),
                # użyj nastroju określonego na podstawie mimiki
                if mood not in ["happy", "sad", "neutral", "surprised", "wink"]:
                    mood = current_mood

                # Rysujemy twarz z uwzględnieniem wyrazu mimiki
                self._draw_face_with_expressions(
                    canvas, head_center, head_radius, smoothed_expressions, mood
                )
            else:
                # W przeciwnym razie rysujemy prostą twarz w zależności od nastroju
                self._draw_mood_face(canvas, head_center, head_radius, mood)

            # Dodaj logowanie co 500 klatek
            if self.frame_count % 500 == 0:
                self.logger.debug(
                    "SimpleFaceRenderer",
                    f"Wyrenderowano {self.frame_count} klatek twarzy, aktualny nastrój: {mood}",
                    log_type="DRAWING",
                )

        except Exception as e:
            self.logger.error(
                "SimpleFaceRenderer", f"Błąd podczas rysowania twarzy: {str(e)}", log_type="DRAWING"
            )

    def _average_expressions(self) -> Dict[str, float]:
        """
        Uśrednia wyrazy twarzy z historii.

        Returns:
            Dict[str, float]: Uśrednione wartości
        """
        if not self.expressions_history:
            return self.last_expressions

        # Inicjalizacja słownika z kluczami wszystkich wyrażeń
        avg_expressions = {key: 0.0 for key in self.last_expressions.keys()}

        # Sumowanie wszystkich wartości
        for expressions in self.expressions_history:
            for key in avg_expressions.keys():
                if key in expressions:
                    avg_expressions[key] += expressions[key]

        # Dzielenie przez ilość próbek
        history_size = len(self.expressions_history)
        for key in avg_expressions.keys():
            avg_expressions[key] /= history_size

        return avg_expressions

    def _determine_mood_from_smile(self, smile_value: float) -> str:
        """
        Określa nastrój na podstawie wartości uśmiechu z histerezą.

        Args:
            smile_value (float): Wartość uśmiechu (0.0-1.0)

        Returns:
            str: Nastrój ("happy", "sad", "neutral")
        """
        # Obecny nastrój bazujący na ostatniej wartości smile
        current_mood = "neutral"

        if self.last_expressions["smile"] > self.neutral_upper + self.smile_histeresis:
            current_mood = "happy"
        elif self.last_expressions["smile"] < self.neutral_lower - self.smile_histeresis:
            current_mood = "sad"

        # Sprawdzamy czy wartość uśmiechu znacząco się zmieniła
        if smile_value > self.neutral_upper + self.smile_histeresis:
            # Wyraźny uśmiech
            current_mood = "happy"
        elif smile_value < self.neutral_lower - self.smile_histeresis:
            # Wyraźny smutek
            current_mood = "sad"
        elif self.neutral_lower <= smile_value <= self.neutral_upper:
            # Neutralna mina
            current_mood = "neutral"

        return current_mood

    def _smooth_expressions(self, expressions: Dict[str, float]) -> Dict[str, float]:
        """
        Wygładza wyrazy twarzy między klatkami.

        Args:
            expressions (Dict[str, float]): Nowe wartości wyrazów twarzy

        Returns:
            Dict[str, float]: Wygładzone wartości
        """
        smoothed = {}

        for key, value in expressions.items():
            if key in self.last_expressions:
                # Wygładź wyrazy twarzy używając smooth_factor
                smoothed[key] = self.last_expressions[key] * self.smooth_factor + value * (
                    1 - self.smooth_factor
                )
            else:
                smoothed[key] = value

        # Zapisz do następnej klatki
        self.last_expressions = smoothed

        return smoothed

    def _draw_face_with_expressions(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        expressions: Dict[str, float],
        mood: str,
    ) -> None:
        """
        Rysuje twarz na podstawie wartości wyrazów twarzy.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            center (Tuple[int, int]): Środek głowy (x, y)
            radius (int): Promień głowy
            expressions (Dict[str, float]): Wartości wyrazów twarzy
            mood (str): Obecny nastrój twarzy ("happy", "sad", "neutral")
        """
        # Środek głowy
        cx, cy = center

        # Pobieramy wartości wyrazów twarzy
        mouth_open = expressions.get("mouth_open", 0.0)
        smile = expressions.get("smile", 0.5)  # Domyślny uśmiech
        left_eye_open = expressions.get("left_eye_open", 1.0)
        right_eye_open = expressions.get("right_eye_open", 1.0)

        # Rysujemy oczy
        eye_offset_x = int(radius * 0.3)
        eye_offset_y = int(radius * 0.2)
        eye_size = max(2, int(radius * 0.1))

        # Lewe oko
        left_eye_pos = (cx - eye_offset_x, cy - eye_offset_y)
        if left_eye_open > 0.3:
            cv2.circle(canvas, left_eye_pos, eye_size, self.feature_color, -1)
        else:
            cv2.line(
                canvas,
                (left_eye_pos[0] - eye_size, left_eye_pos[1]),
                (left_eye_pos[0] + eye_size, left_eye_pos[1]),
                self.feature_color,
                2,
            )

        # Prawe oko
        right_eye_pos = (cx + eye_offset_x, cy - eye_offset_y)
        if right_eye_open > 0.3:
            cv2.circle(canvas, right_eye_pos, eye_size, self.feature_color, -1)
        else:
            cv2.line(
                canvas,
                (right_eye_pos[0] - eye_size, right_eye_pos[1]),
                (right_eye_pos[0] + eye_size, right_eye_pos[1]),
                self.feature_color,
                2,
            )

        # Rysujemy usta
        mouth_y = cy + int(radius * 0.2)
        mouth_width = int(radius * 0.6)
        mouth_height = int(radius * 0.3)

        # Logowanie co 100 klatek
        if self.frame_count % 100 == 0:
            self.logger.debug(
                "SimpleFaceRenderer",
                f"Wyrazy twarzy: smile={smile:.2f}, mouth_open={mouth_open:.2f}, mood={mood}",
                log_type="DRAWING",
            )

        if mouth_open > 0.2:
            # Otwarte usta - elipsa
            mouth_open_height = int(mouth_height * min(2.5, 1.0 + mouth_open * 2))
            cv2.ellipse(
                canvas,
                (cx, mouth_y),
                (mouth_width, mouth_open_height),
                0,  # kąt
                0,
                360,  # pełna elipsa
                self.feature_color,
                2,  # grubość
            )
        else:
            # Zamknięte usta - różne rodzaje w zależności od nastroju
            if mood == "happy":
                # Uśmiech - łuk w dół
                cv2.ellipse(
                    canvas,
                    (cx, mouth_y),
                    (mouth_width, mouth_height),
                    0,  # kąt
                    0,
                    180,  # łuk w dół
                    self.feature_color,
                    2,  # grubość
                )
            elif mood == "sad":
                # Smutek - łuk w górę
                cv2.ellipse(
                    canvas,
                    (cx, mouth_y + mouth_height // 2),
                    (mouth_width, mouth_height),
                    0,  # kąt
                    180,
                    360,  # łuk w górę
                    self.feature_color,
                    2,  # grubość
                )
            else:  # Neutralny
                # Neutralny - prosta linia
                cv2.line(
                    canvas,
                    (cx - mouth_width // 2, mouth_y),
                    (cx + mouth_width // 2, mouth_y),
                    self.feature_color,
                    2,  # grubość
                )

    def _draw_mood_face(
        self, canvas: np.ndarray, center: Tuple[int, int], radius: int, mood: str
    ) -> None:
        """
        Rysuje twarz z określonym nastrojem.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            center (Tuple[int, int]): Środek głowy (x, y)
            radius (int): Promień głowy
            mood (str): Nastrój ("happy", "sad", "neutral", "surprised", "wink")
        """
        # Środek głowy
        cx, cy = center

        # Rysujemy oczy
        eye_offset_x = int(radius * 0.3)
        eye_offset_y = int(radius * 0.2)
        eye_size = max(2, int(radius * 0.1))

        # Lewe oko - wyjątek dla mrugnięcia
        left_eye_pos = (cx - eye_offset_x, cy - eye_offset_y)
        if mood == "wink":
            cv2.line(
                canvas,
                (left_eye_pos[0] - eye_size, left_eye_pos[1]),
                (left_eye_pos[0] + eye_size, left_eye_pos[1]),
                self.feature_color,
                2,
            )
        else:
            cv2.circle(canvas, left_eye_pos, eye_size, self.feature_color, -1)

        # Prawe oko
        right_eye_pos = (cx + eye_offset_x, cy - eye_offset_y)
        cv2.circle(canvas, right_eye_pos, eye_size, self.feature_color, -1)

        # Rysujemy usta w zależności od nastroju
        mouth_y = cy + int(radius * 0.2)
        mouth_width = int(radius * 0.6)
        mouth_height = int(radius * 0.3)

        if mood == "happy" or mood == "wink":
            # Uśmiech
            cv2.ellipse(
                canvas,
                (cx, mouth_y),
                (mouth_width, mouth_height),
                0,  # kąt
                0,
                180,  # łuk w dół
                self.feature_color,
                2,  # grubość
            )
        elif mood == "sad":
            # Smutek - odwrócony łuk
            cv2.ellipse(
                canvas,
                (cx, mouth_y + mouth_height // 2),
                (mouth_width, mouth_height),
                0,  # kąt
                180,
                360,  # łuk w górę
                self.feature_color,
                2,  # grubość
            )
        elif mood == "surprised":
            # Zaskoczenie - otwarte usta (okrąg)
            cv2.circle(
                canvas, (cx, mouth_y), int(mouth_width * 0.4), self.feature_color, 2  # grubość
            )
        else:  # neutral
            # Neutralny - prosta linia
            cv2.line(
                canvas,
                (cx - mouth_width // 2, mouth_y),
                (cx + mouth_width // 2, mouth_y),
                self.feature_color,
                2,  # grubość
            )

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan renderera.
        """
        self.last_expressions = {
            "mouth_open": 0.0,
            "smile": 0.5,  # Neutralna wartość
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0,
        }
        self.expressions_history = []
        self.frame_count = 0

        self.logger.debug(
            "SimpleFaceRenderer", "Zresetowano stan renderera twarzy", log_type="DRAWING"
        )
