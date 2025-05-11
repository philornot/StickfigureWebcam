#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/stick_figure_renderer.py

import math
import time
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger
from .face_renderer import SimpleFaceRenderer


class StickFigureRenderer:
    """
    Klasa do renderowania stick figure (patyczaka) z koncentracją na popierściu.
    Zapewnia płynne animacje rąk, nawet gdy nie są one wykryte przez kamerę.
    """

    # Indeksy punktów z MediaPipe FaceMesh i Pose (używane do identyfikacji)
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(
        self,
        canvas_width: int = 640,
        canvas_height: int = 480,
        line_thickness: int = 3,
        head_radius_factor: float = 0.075,
        bg_color: Tuple[int, int, int] = (255, 255, 255),  # Białe tło
        figure_color: Tuple[int, int, int] = (0, 0, 0),  # Czarny patyczak
        smooth_factor: float = 0.3,  # Współczynnik wygładzania ruchu
        smoothing_history: int = 3,  # Liczba klatek historii do wygładzania
        logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja renderera stick figure.

        Args:
            canvas_width (int): Szerokość obszaru rysowania
            canvas_height (int): Wysokość obszaru rysowania
            line_thickness (int): Grubość linii stick figure
            head_radius_factor (float): Promień głowy jako ułamek wysokości
            bg_color (Tuple[int, int, int]): Kolor tła (BGR)
            figure_color (Tuple[int, int, int]): Kolor stick figure (BGR)
            smooth_factor (float): Współczynnik wygładzania ruchu (0.0-1.0)
            smoothing_history (int): Liczba klatek historii do wygładzania
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        # Inicjalizacja loggera
        self.logger = logger or CustomLogger()

        # Parametry renderowania
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.line_thickness = max(2, line_thickness)
        self.head_radius_factor = head_radius_factor
        self.bg_color = bg_color
        self.figure_color = figure_color
        self.smooth_factor = smooth_factor
        self.smoothing_history = smoothing_history

        # Obliczanie promienia głowy
        self.head_radius = int(head_radius_factor * canvas_height)

        # Inicjalizacja renderera twarzy
        self.face_renderer = SimpleFaceRenderer(
            feature_color=figure_color,
            smooth_factor=smooth_factor
        )

        # Stan nastroju
        self.mood = "happy"  # Domyślny nastrój

        # Pozycja zawsze na środku ekranu
        self.head_center = (canvas_width // 2, canvas_height // 3)

        # Historia detekcji punktów - dla wygładzania
        self.landmark_history = []

        # Licznik klatek - do animacji
        self.frame_count = 0
        self.animation_start_time = time.time()

        # Ostatnio wykryte pozycje rąk
        self.last_left_elbow = None
        self.last_right_elbow = None
        self.last_left_wrist = None
        self.last_right_wrist = None

        # Flagi do śledzenia widoczności
        self.left_arm_visible = False
        self.right_arm_visible = False
        self.left_arm_visibility_time = 0
        self.right_arm_visibility_time = 0

        # Parametry animacji dla rąk
        self.arms_animation_speed = 1.0  # Podstawowa prędkość animacji
        self.arms_animation_range = 10  # Zakres ruchu przy idle animation

        self.logger.info(
            "StickFigureRenderer",
            f"Zainicjalizowano renderer stick figure ({canvas_width}x{canvas_height})",
            log_type="DRAWING"
        )

    def render(self, face_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Renderuje stick figure na środku ekranu.

        Jeśli dostępne są dane punktów charakterystycznych twarzy, używa ich do
        animacji twarzy i rąk. W przeciwnym razie animuje postać w sposób domyślny.

        Args:
            face_data (Optional[Dict[str, Any]]): Dane twarzy z detektora

        Returns:
            np.ndarray: Obraz z narysowanym stick figure
        """
        self.frame_count += 1

        try:
            # Tworzenie pustego obrazu
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas[:] = self.bg_color

            # Aktualizacja pozycji rąk na podstawie danych twarzy (jeśli dostępne)
            self._update_arm_positions(face_data)

            # Rysowanie stick figure
            self._draw_figure(canvas)

            # Rysowanie twarzy
            self.face_renderer.draw_face(
                canvas,
                self.head_center,
                self.head_radius,
                self.mood,
                face_data
            )

            # Logowanie co 300 klatek lub gdy debugowanie jest aktywne
            if self.frame_count % 300 == 0:
                self.logger.debug(
                    "StickFigureRenderer",
                    f"Wyrenderowano {self.frame_count} klatek. Widoczność rąk: "
                    f"L={self.left_arm_visible}, P={self.right_arm_visible}",
                    log_type="DRAWING"
                )

            return canvas

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas renderowania stick figure: {str(e)}",
                log_type="DRAWING"
            )
            # W przypadku błędu zwracamy pusty biały obraz
            return np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

    def _update_arm_positions(self, face_data: Optional[Dict[str, Any]]) -> None:
        """
        Aktualizuje pozycje rąk na podstawie danych face_data lub animacji idle.

        Jeśli ręce są widoczne w danych wejściowych, aktualizuje ich pozycje.
        W przeciwnym razie stopniowo wraca do pozycji neutralnej z animacją.

        Args:
            face_data (Optional[Dict[str, Any]]): Dane z detektora twarzy
        """
        try:
            current_time = time.time()

            # Środek głowy - punkt odniesienia dla całej postaci
            head_x, head_y = self.head_center

            # Pozycje barków - zależne od środka głowy
            shoulder_y = head_y + self.head_radius + int(self.head_radius * 0.5)
            shoulder_width = int(self.head_radius * 3)
            left_shoulder_x = head_x - shoulder_width // 2
            right_shoulder_x = head_x + shoulder_width // 2

            # Sprawdzenie czy mamy dane mediapipe
            have_landmarks = False
            landmarks = None

            if face_data and "landmarks" in face_data and face_data["landmarks"]:
                landmarks = face_data["landmarks"]
                have_landmarks = True

            # Parametry animacji idle dla rąk
            animation_time = current_time - self.animation_start_time
            idle_animation_factor = math.sin(animation_time * 0.5) * 0.5 + 0.5  # Wartość 0-1

            # Neutralne pozycje łokci (gdy nie ma detekcji)
            neutral_elbow_offset_y = int(self.head_radius * 1.0)
            neutral_elbow_offset_x = int(self.head_radius * 0.7)

            # Neutralne pozycje nadgarstków
            neutral_wrist_offset_y = int(self.head_radius * 1.0)
            neutral_wrist_offset_x = int(self.head_radius * 0.5)

            # Dodanie delikatnej animacji do pozycji neutralnej
            idle_animation_amount = self.arms_animation_range * idle_animation_factor

            # LEWA RĘKA - aktualizacja
            # Sprawdzenie czy mamy dane dla lewej ręki
            left_elbow_detected = False
            left_wrist_detected = False

            if have_landmarks and len(landmarks) > self.LEFT_ELBOW and len(landmarks) > self.LEFT_WRIST:
                # Sprawdzenie widoczności łokcia
                if landmarks[self.LEFT_ELBOW][3] > 0.5:  # Jeśli widoczność powyżej 0.5
                    # Wykryto lewy łokieć
                    left_elbow_detected = True
                    canvas_x = int(landmarks[self.LEFT_ELBOW][0] * self.canvas_width)
                    canvas_y = int(landmarks[self.LEFT_ELBOW][1] * self.canvas_height)

                    # Jeśli to pierwszy raz, gdy wykryto łokieć - zapisz czas
                    if not self.left_arm_visible:
                        self.left_arm_visibility_time = current_time
                        self.left_arm_visible = True
                        self.logger.debug(
                            "StickFigureRenderer",
                            "Wykryto lewą rękę - rozpoczęcie śledzenia",
                            log_type="DRAWING"
                        )

                    # Płynne przejście do wykrytej pozycji
                    if self.last_left_elbow is None:
                        self.last_left_elbow = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(self.last_left_elbow[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_left_elbow[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_left_elbow = (new_x, new_y)

                # Sprawdzenie widoczności nadgarstka
                if landmarks[self.LEFT_WRIST][3] > 0.5:  # Jeśli widoczność powyżej 0.5
                    # Wykryto lewy nadgarstek
                    left_wrist_detected = True
                    canvas_x = int(landmarks[self.LEFT_WRIST][0] * self.canvas_width)
                    canvas_y = int(landmarks[self.LEFT_WRIST][1] * self.canvas_height)

                    # Płynne przejście do wykrytej pozycji
                    if self.last_left_wrist is None:
                        self.last_left_wrist = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(self.last_left_wrist[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_left_wrist[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_left_wrist = (new_x, new_y)

            # Jeśli nie wykryto lewej ręki
            if not left_elbow_detected or not left_wrist_detected:
                # Jeśli wcześniej była widoczna, oznacz jako niewidoczną i zapisz czas
                if self.left_arm_visible:
                    self.left_arm_visible = False
                    self.left_arm_visibility_time = current_time
                    self.logger.debug(
                        "StickFigureRenderer",
                        "Utracono śledzenie lewej ręki - powrót do animacji domyślnej",
                        log_type="DRAWING"
                    )

                # Czas od utraty widoczności
                time_since_lost = current_time - self.left_arm_visibility_time
                transition_factor = min(1.0, time_since_lost * 2.0)  # Pełna animacja po 0.5s

                # Neutralna pozycja lewego łokcia z animacją
                neutral_left_elbow_x = left_shoulder_x - neutral_elbow_offset_x
                neutral_left_elbow_y = shoulder_y + neutral_elbow_offset_y + int(idle_animation_amount)

                # Neutralna pozycja lewego nadgarstka z animacją
                neutral_left_wrist_x = neutral_left_elbow_x - neutral_wrist_offset_x
                neutral_left_wrist_y = neutral_left_elbow_y + neutral_wrist_offset_y - int(idle_animation_amount)

                # Jeśli mamy zapisane ostatnie pozycje, płynnie przechodzimy do pozycji neutralnej
                if self.last_left_elbow is not None:
                    # Interpolacja liniowa między ostatnią wykrytą pozycją a pozycją neutralną
                    new_x = int(
                        self.last_left_elbow[0] * (1 - transition_factor) + neutral_left_elbow_x * transition_factor)
                    new_y = int(
                        self.last_left_elbow[1] * (1 - transition_factor) + neutral_left_elbow_y * transition_factor)
                    self.last_left_elbow = (new_x, new_y)
                else:
                    # Jeśli nie mamy historii, używamy bezpośrednio pozycji neutralnej
                    self.last_left_elbow = (neutral_left_elbow_x, neutral_left_elbow_y)

                if self.last_left_wrist is not None:
                    # Interpolacja liniowa między ostatnią wykrytą pozycją a pozycją neutralną
                    new_x = int(
                        self.last_left_wrist[0] * (1 - transition_factor) + neutral_left_wrist_x * transition_factor)
                    new_y = int(
                        self.last_left_wrist[1] * (1 - transition_factor) + neutral_left_wrist_y * transition_factor)
                    self.last_left_wrist = (new_x, new_y)
                else:
                    # Jeśli nie mamy historii, używamy bezpośrednio pozycji neutralnej
                    self.last_left_wrist = (neutral_left_wrist_x, neutral_left_wrist_y)

            # PRAWA RĘKA - aktualizacja (analogicznie jak lewa)
            # Sprawdzenie czy mamy dane dla prawej ręki
            right_elbow_detected = False
            right_wrist_detected = False

            if have_landmarks and len(landmarks) > self.RIGHT_ELBOW and len(landmarks) > self.RIGHT_WRIST:
                # Sprawdzenie widoczności łokcia
                if landmarks[self.RIGHT_ELBOW][3] > 0.5:  # Jeśli widoczność powyżej 0.5
                    # Wykryto prawy łokieć
                    right_elbow_detected = True
                    canvas_x = int(landmarks[self.RIGHT_ELBOW][0] * self.canvas_width)
                    canvas_y = int(landmarks[self.RIGHT_ELBOW][1] * self.canvas_height)

                    # Jeśli to pierwszy raz, gdy wykryto łokieć - zapisz czas
                    if not self.right_arm_visible:
                        self.right_arm_visibility_time = current_time
                        self.right_arm_visible = True
                        self.logger.debug(
                            "StickFigureRenderer",
                            "Wykryto prawą rękę - rozpoczęcie śledzenia",
                            log_type="DRAWING"
                        )

                    # Płynne przejście do wykrytej pozycji
                    if self.last_right_elbow is None:
                        self.last_right_elbow = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(self.last_right_elbow[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_right_elbow[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_right_elbow = (new_x, new_y)

                # Sprawdzenie widoczności nadgarstka
                if landmarks[self.RIGHT_WRIST][3] > 0.5:  # Jeśli widoczność powyżej 0.5
                    # Wykryto prawy nadgarstek
                    right_wrist_detected = True
                    canvas_x = int(landmarks[self.RIGHT_WRIST][0] * self.canvas_width)
                    canvas_y = int(landmarks[self.RIGHT_WRIST][1] * self.canvas_height)

                    # Płynne przejście do wykrytej pozycji
                    if self.last_right_wrist is None:
                        self.last_right_wrist = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(self.last_right_wrist[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_right_wrist[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_right_wrist = (new_x, new_y)

            # Jeśli nie wykryto prawej ręki
            if not right_elbow_detected or not right_wrist_detected:
                # Jeśli wcześniej była widoczna, oznacz jako niewidoczną i zapisz czas
                if self.right_arm_visible:
                    self.right_arm_visible = False
                    self.right_arm_visibility_time = current_time
                    self.logger.debug(
                        "StickFigureRenderer",
                        "Utracono śledzenie prawej ręki - powrót do animacji domyślnej",
                        log_type="DRAWING"
                    )

                # Czas od utraty widoczności
                time_since_lost = current_time - self.right_arm_visibility_time
                transition_factor = min(1.0, time_since_lost * 2.0)  # Pełna animacja po 0.5s

                # Neutralna pozycja prawego łokcia z animacją
                neutral_right_elbow_x = right_shoulder_x + neutral_elbow_offset_x
                neutral_right_elbow_y = shoulder_y + neutral_elbow_offset_y + int(idle_animation_amount)

                # Neutralna pozycja prawego nadgarstka z animacją
                neutral_right_wrist_x = neutral_right_elbow_x + neutral_wrist_offset_x
                neutral_right_wrist_y = neutral_right_elbow_y + neutral_wrist_offset_y - int(idle_animation_amount)

                # Jeśli mamy zapisane ostatnie pozycje, płynnie przechodzimy do pozycji neutralnej
                if self.last_right_elbow is not None:
                    # Interpolacja liniowa między ostatnią wykrytą pozycją a pozycją neutralną
                    new_x = int(
                        self.last_right_elbow[0] * (1 - transition_factor) + neutral_right_elbow_x * transition_factor)
                    new_y = int(
                        self.last_right_elbow[1] * (1 - transition_factor) + neutral_right_elbow_y * transition_factor)
                    self.last_right_elbow = (new_x, new_y)
                else:
                    # Jeśli nie mamy historii, używamy bezpośrednio pozycji neutralnej
                    self.last_right_elbow = (neutral_right_elbow_x, neutral_right_elbow_y)

                if self.last_right_wrist is not None:
                    # Interpolacja liniowa między ostatnią wykrytą pozycją a pozycją neutralną
                    new_x = int(
                        self.last_right_wrist[0] * (1 - transition_factor) + neutral_right_wrist_x * transition_factor)
                    new_y = int(
                        self.last_right_wrist[1] * (1 - transition_factor) + neutral_right_wrist_y * transition_factor)
                    self.last_right_wrist = (new_x, new_y)
                else:
                    # Jeśli nie mamy historii, używamy bezpośrednio pozycji neutralnej
                    self.last_right_wrist = (neutral_right_wrist_x, neutral_right_wrist_y)

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas aktualizacji pozycji rąk: {str(e)}",
                log_type="DRAWING"
            )

    def _draw_figure(self, canvas: np.ndarray) -> None:
        """
        Rysuje stick figure względem pozycji głowy z animowanymi rękoma.

        Args:
            canvas (np.ndarray): Płótno do rysowania
        """
        try:
            # 1. Głowa - okrąg
            cv2.circle(canvas, self.head_center, self.head_radius, self.figure_color, self.line_thickness)

            # 2. Obliczenie pozycji innych części ciała względem głowy
            # Środek głowy
            head_x, head_y = self.head_center

            # Klatka piersiowa - poniżej głowy
            torso_top_y = head_y + self.head_radius + 10
            torso_length = int(self.head_radius * 2.5)
            torso_bottom_y = torso_top_y + torso_length

            # Ramiona - powyżej klatki piersiowej
            shoulder_y = torso_top_y + int(self.head_radius * 0.5)
            shoulder_width = int(self.head_radius * 3)
            left_shoulder_x = head_x - shoulder_width // 2
            right_shoulder_x = head_x + shoulder_width // 2

            # 3. Rysowanie tułowia (linia pionowa od głowy w dół)
            cv2.line(
                canvas,
                (head_x, torso_top_y),
                (head_x, torso_bottom_y),
                self.figure_color,
                self.line_thickness
            )

            # 4. Rysowanie linii barków
            cv2.line(
                canvas,
                (left_shoulder_x, shoulder_y),
                (right_shoulder_x, shoulder_y),
                self.figure_color,
                self.line_thickness
            )

            # 5. Rysowanie rąk na podstawie obliczonych wcześniej pozycji

            # Sprawdzenie czy mamy zapisane pozycje rąk
            if self.last_left_elbow is not None and self.last_left_wrist is not None:
                # Rysowanie lewego ramienia (shoulder -> elbow)
                cv2.line(
                    canvas,
                    (left_shoulder_x, shoulder_y),
                    self.last_left_elbow,
                    self.figure_color,
                    self.line_thickness
                )

                # Rysowanie lewego przedramienia (elbow -> wrist)
                cv2.line(
                    canvas,
                    self.last_left_elbow,
                    self.last_left_wrist,
                    self.figure_color,
                    self.line_thickness
                )

            if self.last_right_elbow is not None and self.last_right_wrist is not None:
                # Rysowanie prawego ramienia (shoulder -> elbow)
                cv2.line(
                    canvas,
                    (right_shoulder_x, shoulder_y),
                    self.last_right_elbow,
                    self.figure_color,
                    self.line_thickness
                )

                # Rysowanie prawego przedramienia (elbow -> wrist)
                cv2.line(
                    canvas,
                    self.last_right_elbow,
                    self.last_right_wrist,
                    self.figure_color,
                    self.line_thickness
                )

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas rysowania postaci: {str(e)}",
                log_type="DRAWING"
            )

    def set_colors(
        self,
        bg_color: Optional[Tuple[int, int, int]] = None,
        figure_color: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """
        Aktualizuje kolory używane do rysowania.

        Args:
            bg_color (Optional[Tuple[int, int, int]]): Nowy kolor tła (BGR)
            figure_color (Optional[Tuple[int, int, int]]): Nowy kolor stick figure (BGR)
        """
        if bg_color is not None:
            self.bg_color = bg_color
            self.logger.debug(
                "StickFigureRenderer",
                f"Zmieniono kolor tła na {bg_color}",
                log_type="DRAWING"
            )

        if figure_color is not None:
            self.figure_color = figure_color
            # Aktualizacja koloru w rendererze twarzy
            self.face_renderer.feature_color = figure_color
            self.logger.debug(
                "StickFigureRenderer",
                f"Zmieniono kolor figury na {figure_color}",
                log_type="DRAWING"
            )

    def set_mood(self, mood: str) -> None:
        """
        Ustawia nastrój stick figure, który wpływa na mimikę twarzy.

        Args:
            mood (str): Nastrój: "happy", "sad", "neutral", "surprised", "wink"
        """
        valid_moods = ["happy", "sad", "neutral", "surprised", "wink"]
        if mood in valid_moods:
            self.mood = mood
            self.logger.info(
                "StickFigureRenderer",
                f"Zmieniono nastrój na: {mood}",
                log_type="DRAWING"
            )
        else:
            self.logger.warning(
                "StickFigureRenderer",
                f"Nieprawidłowy nastrój: {mood}. Dozwolone wartości: {valid_moods}",
                log_type="DRAWING"
            )

    def set_line_thickness(self, thickness: int) -> None:
        """
        Aktualizuje grubość linii.

        Args:
            thickness (int): Nowa grubość linii
        """
        self.line_thickness = max(1, thickness)
        self.logger.debug(
            "StickFigureRenderer",
            f"Zmieniono grubość linii na {self.line_thickness}",
            log_type="DRAWING"
        )

    def set_smoothing(self, smooth_factor: float, history_length: int) -> None:
        """
        Aktualizuje parametry wygładzania.

        Args:
            smooth_factor (float): Nowy współczynnik wygładzania (0.0-1.0)
            history_length (int): Nowa długość historii do wygładzania
        """
        self.smooth_factor = max(0.0, min(1.0, smooth_factor))
        self.smoothing_history = max(1, history_length)
        self.logger.debug(
            "StickFigureRenderer",
            f"Zaktualizowano parametry wygładzania: faktor={self.smooth_factor}, "
            f"długość historii={self.smoothing_history}",
            log_type="DRAWING"
        )

    def resize(self, width: int, height: int) -> None:
        """
        Zmienia rozmiar płótna.

        Args:
            width (int): Nowa szerokość
            height (int): Nowa wysokość
        """
        self.canvas_width = width
        self.canvas_height = height

        # Aktualizacja promienia głowy
        self.head_radius = int(self.head_radius_factor * height)

        # Aktualizacja pozycji głowy na środku ekranu
        self.head_center = (width // 2, height // 3)

        # Reset historii - bo zmieniamy skalę
        self.landmark_history = []
        self.last_left_elbow = None
        self.last_right_elbow = None
        self.last_left_wrist = None
        self.last_right_wrist = None

        self.logger.info(
            "StickFigureRenderer",
            f"Zmieniono rozmiar płótna na {width}x{height}",
            log_type="DRAWING"
        )

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan renderera.
        """
        self.mood = "happy"  # Przywrócenie domyślnego nastroju
        self.landmark_history = []
        self.last_left_elbow = None
        self.last_right_elbow = None
        self.last_left_wrist = None
        self.last_right_wrist = None
        self.left_arm_visible = False
        self.right_arm_visible = False
        self.animation_start_time = time.time()
        self.face_renderer.reset()

        self.logger.info(
            "StickFigureRenderer",
            "Zresetowano stan renderera",
            log_type="DRAWING"
        )
