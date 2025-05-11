#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/stick_figure_renderer.py

import math
import time
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger
from .face_renderer import SimpleFaceRenderer
from .pose_analyzer import PoseAnalyzer


class StickFigureRenderer:
    """
    Klasa do renderowania stick figure (patyczaka) z koncentracją na popierściu.
    Zapewnia płynne animacje rąk, z dokładniejszym wykrywaniem i renderowaniem
    ramion zamiast samych dłoni.
    """

    # Indeksy punktów z MediaPipe FaceMesh i Pose
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
        line_thickness: int = 4,  # Zwiększona grubość linii
        head_radius_factor: float = 0.12,  # Zwiększony promień głowy
        bg_color: Tuple[int, int, int] = (255, 255, 255),  # Białe tło
        figure_color: Tuple[int, int, int] = (0, 0, 0),  # Czarny patyczak
        chair_color: Tuple[int, int, int] = (150, 75, 0),
        # Kolor krzesła (niewykorzystywany, ale pozostawiony dla kompatybilności)
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
            chair_color (Tuple[int, int, int]): Kolor krzesła (BGR) - niewykorzystywany, zachowany dla kompatybilności
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
        self.chair_color = chair_color  # Zachowany dla kompatybilności
        self.smooth_factor = smooth_factor
        self.smoothing_history = smoothing_history

        # Obliczanie promienia głowy
        self.head_radius = int(head_radius_factor * canvas_height)

        # Inicjalizacja renderera twarzy
        self.face_renderer = SimpleFaceRenderer(
            feature_color=figure_color,
            smooth_factor=smooth_factor,
            logger=self.logger
        )

        # Inicjalizacja analizatora pozy do analizy górnej części ciała
        self.pose_analyzer = PoseAnalyzer(logger=self.logger)

        # Stan nastroju
        self.mood = "happy"  # Domyślny nastrój

        # Pozycja głowy - teraz wyżej (1/4 zamiast 1/3 wysokości)
        self.head_center = (canvas_width // 2, canvas_height // 4)

        # Historia detekcji punktów - dla wygładzania
        self.landmark_history: List = []

        # Licznik klatek - do animacji
        self.frame_count = 0
        self.animation_start_time = time.time()

        # Ostatnio wykryte pozycje ramion i rąk
        self.last_left_shoulder = None
        self.last_right_shoulder = None
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
        self.arms_animation_speed = 0.8  # Spowolniona prędkość animacji
        self.arms_animation_range = 15  # Zwiększony zakres ruchu przy idle animation

        # Parametry proporcji popiersia
        self.torso_length_factor = 1.8  # Długość torsu jako mnożnik promienia głowy
        self.shoulder_width_factor = 3.2  # Szerokość ramion jako mnożnik promienia głowy

        self.logger.info(
            "StickFigureRenderer",
            f"Zainicjalizowano renderer stick figure ({canvas_width}x{canvas_height}) "
            f"z większym popierściem (head_radius={self.head_radius})",
            log_type="DRAWING"
        )

    def render(self, face_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Renderuje stick figure koncentrując się na popierściu (górna część ciała).

        Jeśli dostępne są dane punktów charakterystycznych twarzy lub rąk, używa ich do
        animacji twarzy i rąk. W przeciwnym razie animuje postać w sposób domyślny.

        Args:
            face_data (Optional[Dict[str, Any]]): Dane twarzy i rąk z detektora

        Returns:
            np.ndarray: Obraz z narysowanym stick figure
        """
        self.frame_count += 1

        try:
            # Tworzenie pustego obrazu
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas[:] = self.bg_color

            # Aktualizacja pozycji rąk na podstawie danych (jeśli dostępne)
            self._update_arm_positions(face_data)

            # Rysowanie stick figure - tylko popiersie
            self._draw_upper_body(canvas)

            # Rysowanie twarzy
            self.face_renderer.draw_face(
                canvas,
                self.head_center,
                self.head_radius,
                self.mood,
                face_data
            )

            # Logowanie co 300 klatek
            if self.frame_count % 300 == 0:
                self.logger.debug(
                    "StickFigureRenderer",
                    f"Wyrenderowano {self.frame_count} klatek. Widoczność ramion: "
                    f"L={self.left_arm_visible}, P={self.right_arm_visible}",
                    log_type="DRAWING"
                )

            return canvas

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas renderowania stick figure: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)}
            )
            # W przypadku błędu zwracamy pusty biały obraz
            return np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

    def _update_arm_positions(self, face_data: Optional[Dict[str, Any]]) -> None:
        """
        Aktualizuje pozycje ramion i rąk na podstawie danych face_data lub animacji idle.

        Poszerzona wersja uwzględniająca wykrywanie ramion (barków).

        Args:
            face_data (Optional[Dict[str, Any]]): Dane z detektora twarzy i rąk
        """
        try:
            current_time = time.time()

            # Środek głowy - punkt odniesienia dla całej postaci
            head_x, head_y = self.head_center

            # Pozycje barków - zależne od środka głowy
            shoulder_y = head_y + self.head_radius + int(self.head_radius * 0.3)
            shoulder_width = int(self.head_radius * self.shoulder_width_factor)
            left_shoulder_x = head_x - shoulder_width // 2
            right_shoulder_x = head_x + shoulder_width // 2

            # Domyślne pozycje barków
            default_left_shoulder = (left_shoulder_x, shoulder_y)
            default_right_shoulder = (right_shoulder_x, shoulder_y)

            # Sprawdzenie czy mamy dane mediapipe
            have_landmarks = False
            landmarks = None
            upper_body_data = None

            if face_data and "landmarks" in face_data and face_data["landmarks"]:
                landmarks = face_data["landmarks"]
                have_landmarks = True

                # Analiza górnej części ciała, jeśli mamy dane punktów
                if len(landmarks) >= 17:  # Potrzebujemy punktów do nadgarstków
                    upper_body_data = self.pose_analyzer.analyze_upper_body(
                        landmarks, self.canvas_width, self.canvas_height
                    )

            # Parametry animacji idle dla rąk
            animation_time = current_time - self.animation_start_time
            idle_animation_factor = math.sin(animation_time * self.arms_animation_speed) * 0.5 + 0.5  # Wartość 0-1

            # Neutralne pozycje łokci (gdy nie ma detekcji)
            neutral_elbow_offset_y = int(self.head_radius * 0.8)
            neutral_elbow_offset_x = int(self.head_radius * 0.7)

            # Neutralne pozycje nadgarstków
            neutral_wrist_offset_y = int(self.head_radius * 0.8)
            neutral_wrist_offset_x = int(self.head_radius * 0.5)

            # Dodanie delikatnej animacji do pozycji neutralnej
            idle_animation_amount = self.arms_animation_range * idle_animation_factor

            # Sprawdzenie danych z hands_data (kompatybilność z poprzednią wersją)
            hands_data = None
            if face_data and "hands_data" in face_data:
                hands_data = face_data["hands_data"]

            # ===== AKTUALIZACJA POZYCJI BARKÓW =====
            left_shoulder_detected = False
            right_shoulder_detected = False

            # Sprawdzenie czy mamy dane o barkach z analizy górnej części ciała
            if upper_body_data and upper_body_data["has_shoulders"]:
                shoulders = upper_body_data["shoulder_positions"]
                if shoulders and shoulders[0]:  # Lewy bark
                    left_shoulder_detected = True
                    canvas_x, canvas_y = shoulders[0]

                    # Płynne przejście do wykrytej pozycji
                    if self.last_left_shoulder is None:
                        self.last_left_shoulder = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(
                            self.last_left_shoulder[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(
                            self.last_left_shoulder[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_left_shoulder = (new_x, new_y)

                if shoulders and shoulders[1]:  # Prawy bark
                    right_shoulder_detected = True
                    canvas_x, canvas_y = shoulders[1]

                    # Płynne przejście do wykrytej pozycji
                    if self.last_right_shoulder is None:
                        self.last_right_shoulder = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(
                            self.last_right_shoulder[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(
                            self.last_right_shoulder[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_right_shoulder = (new_x, new_y)

            # Jeśli nie wykryto lewego barku, używamy domyślnej pozycji
            if not left_shoulder_detected:
                if self.last_left_shoulder is None:
                    self.last_left_shoulder = default_left_shoulder
                else:
                    # Płynne przejście do domyślnej pozycji
                    new_x = int(self.last_left_shoulder[0] * self.smooth_factor + default_left_shoulder[0] * (
                        1 - self.smooth_factor))
                    new_y = int(self.last_left_shoulder[1] * self.smooth_factor + default_left_shoulder[1] * (
                        1 - self.smooth_factor))
                    self.last_left_shoulder = (new_x, new_y)

            # Jeśli nie wykryto prawego barku, używamy domyślnej pozycji
            if not right_shoulder_detected:
                if self.last_right_shoulder is None:
                    self.last_right_shoulder = default_right_shoulder
                else:
                    # Płynne przejście do domyślnej pozycji
                    new_x = int(self.last_right_shoulder[0] * self.smooth_factor + default_right_shoulder[0] * (
                        1 - self.smooth_factor))
                    new_y = int(self.last_right_shoulder[1] * self.smooth_factor + default_right_shoulder[1] * (
                        1 - self.smooth_factor))
                    self.last_right_shoulder = (new_x, new_y)

            # ===== AKTUALIZACJA POZYCJI ŁOKCI =====
            left_elbow_detected = False
            right_elbow_detected = False

            # Najpierw sprawdzamy dane z analizy górnej części ciała
            if upper_body_data and upper_body_data["has_arms"] and upper_body_data["elbow_positions"]:
                elbows = upper_body_data["elbow_positions"]

                # Lewy łokieć
                if elbows[0]:
                    left_elbow_detected = True
                    canvas_x, canvas_y = elbows[0]

                    # Jeśli to pierwszy raz, gdy wykryto łokieć - zapisz czas
                    if not self.left_arm_visible:
                        self.left_arm_visibility_time = current_time
                        self.left_arm_visible = True
                        self.logger.debug(
                            "StickFigureRenderer",
                            "Wykryto lewe ramię - rozpoczęcie śledzenia",
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

                # Prawy łokieć
                if elbows[1]:
                    right_elbow_detected = True
                    canvas_x, canvas_y = elbows[1]

                    # Jeśli to pierwszy raz, gdy wykryto łokieć - zapisz czas
                    if not self.right_arm_visible:
                        self.right_arm_visibility_time = current_time
                        self.right_arm_visible = True
                        self.logger.debug(
                            "StickFigureRenderer",
                            "Wykryto prawe ramię - rozpoczęcie śledzenia",
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

            # Alternatywnie, sprawdzamy dane hands_data (kompatybilność wsteczna)
            elif hands_data and not left_elbow_detected and "left_hand" in hands_data and hands_data["left_hand"]:
                if "elbow" in hands_data["left_hand"] and hands_data["left_hand"]["elbow"]:
                    left_elbow_detected = True
                    elbow_data = hands_data["left_hand"]["elbow"]
                    canvas_x = int(elbow_data[0] * self.canvas_width if elbow_data[0] <= 1.0 else elbow_data[0])
                    canvas_y = int(elbow_data[1] * self.canvas_height if elbow_data[1] <= 1.0 else elbow_data[1])

                    if self.last_left_elbow is None:
                        self.last_left_elbow = (canvas_x, canvas_y)
                    else:
                        new_x = int(self.last_left_elbow[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_left_elbow[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_left_elbow = (new_x, new_y)

            if hands_data and not right_elbow_detected and "right_hand" in hands_data and hands_data["right_hand"]:
                if "elbow" in hands_data["right_hand"] and hands_data["right_hand"]["elbow"]:
                    right_elbow_detected = True
                    elbow_data = hands_data["right_hand"]["elbow"]
                    canvas_x = int(elbow_data[0] * self.canvas_width if elbow_data[0] <= 1.0 else elbow_data[0])
                    canvas_y = int(elbow_data[1] * self.canvas_height if elbow_data[1] <= 1.0 else elbow_data[1])

                    if self.last_right_elbow is None:
                        self.last_right_elbow = (canvas_x, canvas_y)
                    else:
                        new_x = int(self.last_right_elbow[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_right_elbow[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_right_elbow = (new_x, new_y)

            # Jeśli nie wykryto lewego łokcia
            if not left_elbow_detected:
                # Jeśli wcześniej był widoczny, oznacz jako niewidoczny i zapisz czas
                if self.left_arm_visible:
                    self.left_arm_visible = False
                    self.left_arm_visibility_time = current_time
                    self.logger.debug(
                        "StickFigureRenderer",
                        "Utracono śledzenie lewego ramienia - powrót do animacji domyślnej",
                        log_type="DRAWING"
                    )

                # Czas od utraty widoczności
                time_since_lost = current_time - self.left_arm_visibility_time
                transition_factor = min(1.0, time_since_lost * 2.0)  # Pełna animacja po 0.5s

                # Neutralna pozycja lewego łokcia z animacją
                left_shoulder_pos = self.last_left_shoulder or default_left_shoulder
                neutral_left_elbow_x = left_shoulder_pos[0] - neutral_elbow_offset_x
                neutral_left_elbow_y = left_shoulder_pos[1] + neutral_elbow_offset_y + int(idle_animation_amount)

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

            # Jeśli nie wykryto prawego łokcia
            if not right_elbow_detected:
                # Jeśli wcześniej był widoczny, oznacz jako niewidoczny i zapisz czas
                if self.right_arm_visible:
                    self.right_arm_visible = False
                    self.right_arm_visibility_time = current_time
                    self.logger.debug(
                        "StickFigureRenderer",
                        "Utracono śledzenie prawego ramienia - powrót do animacji domyślnej",
                        log_type="DRAWING"
                    )

                # Czas od utraty widoczności
                time_since_lost = current_time - self.right_arm_visibility_time
                transition_factor = min(1.0, time_since_lost * 2.0)  # Pełna animacja po 0.5s

                # Neutralna pozycja prawego łokcia z animacją
                right_shoulder_pos = self.last_right_shoulder or default_right_shoulder
                neutral_right_elbow_x = right_shoulder_pos[0] + neutral_elbow_offset_x
                neutral_right_elbow_y = right_shoulder_pos[1] + neutral_elbow_offset_y + int(idle_animation_amount)

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

            # ===== AKTUALIZACJA POZYCJI NADGARSTKÓW =====
            # To możemy obsłużyć prościej, ponieważ nadgarstki są mniej istotne
            left_wrist_detected = False
            right_wrist_detected = False

            # Najpierw sprawdzamy dane z analizy górnej części ciała
            if upper_body_data and upper_body_data["wrist_positions"]:
                wrists = upper_body_data["wrist_positions"]

                # Lewy nadgarstek
                if wrists[0]:
                    left_wrist_detected = True
                    canvas_x, canvas_y = wrists[0]

                    # Płynne przejście do wykrytej pozycji
                    if self.last_left_wrist is None:
                        self.last_left_wrist = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(self.last_left_wrist[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_left_wrist[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_left_wrist = (new_x, new_y)

                # Prawy nadgarstek
                if wrists[1]:
                    right_wrist_detected = True
                    canvas_x, canvas_y = wrists[1]

                    # Płynne przejście do wykrytej pozycji
                    if self.last_right_wrist is None:
                        self.last_right_wrist = (canvas_x, canvas_y)
                    else:
                        # Wygładzanie ruchu
                        new_x = int(self.last_right_wrist[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_right_wrist[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_right_wrist = (new_x, new_y)

            # Alternatywnie, sprawdzamy dane hands_data (kompatybilność wsteczna)
            elif hands_data and not left_wrist_detected and "left_hand" in hands_data and hands_data["left_hand"]:
                if "wrist" in hands_data["left_hand"] and hands_data["left_hand"]["wrist"]:
                    left_wrist_detected = True
                    wrist_data = hands_data["left_hand"]["wrist"]
                    canvas_x = int(wrist_data[0] * self.canvas_width if wrist_data[0] <= 1.0 else wrist_data[0])
                    canvas_y = int(wrist_data[1] * self.canvas_height if wrist_data[1] <= 1.0 else wrist_data[1])

                    if self.last_left_wrist is None:
                        self.last_left_wrist = (canvas_x, canvas_y)
                    else:
                        new_x = int(self.last_left_wrist[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_left_wrist[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_left_wrist = (new_x, new_y)

            if hands_data and not right_wrist_detected and "right_hand" in hands_data and hands_data["right_hand"]:
                if "wrist" in hands_data["right_hand"] and hands_data["right_hand"]["wrist"]:
                    right_wrist_detected = True
                    wrist_data = hands_data["right_hand"]["wrist"]
                    canvas_x = int(wrist_data[0] * self.canvas_width if wrist_data[0] <= 1.0 else wrist_data[0])
                    canvas_y = int(wrist_data[1] * self.canvas_height if wrist_data[1] <= 1.0 else wrist_data[1])

                    if self.last_right_wrist is None:
                        self.last_right_wrist = (canvas_x, canvas_y)
                    else:
                        new_x = int(self.last_right_wrist[0] * self.smooth_factor + canvas_x * (1 - self.smooth_factor))
                        new_y = int(self.last_right_wrist[1] * self.smooth_factor + canvas_y * (1 - self.smooth_factor))
                        self.last_right_wrist = (new_x, new_y)

            # Jeśli nie wykryto lewego nadgarstka, obliczamy jego pozycję na podstawie łokcia
            if not left_wrist_detected and self.last_left_elbow is not None:
                left_elbow_pos = self.last_left_elbow

                # Pozycja nadgarstka poniżej łokcia - kontynuacja kierunku ramię-łokieć
                elbow_to_wrist_vector_x = -neutral_wrist_offset_x
                elbow_to_wrist_vector_y = neutral_wrist_offset_y - int(idle_animation_amount)

                # Obliczamy pozycję nadgarstka
                left_wrist_x = left_elbow_pos[0] + elbow_to_wrist_vector_x
                left_wrist_y = left_elbow_pos[1] + elbow_to_wrist_vector_y

                # Aktualizacja pozycji nadgarstka z wygładzaniem
                if self.last_left_wrist is None:
                    self.last_left_wrist = (left_wrist_x, left_wrist_y)
                else:
                    new_x = int(self.last_left_wrist[0] * self.smooth_factor + left_wrist_x * (1 - self.smooth_factor))
                    new_y = int(self.last_left_wrist[1] * self.smooth_factor + left_wrist_y * (1 - self.smooth_factor))
                    self.last_left_wrist = (new_x, new_y)

            # Jeśli nie wykryto prawego nadgarstka, obliczamy jego pozycję na podstawie łokcia
            if not right_wrist_detected and self.last_right_elbow is not None:
                right_elbow_pos = self.last_right_elbow

                # Pozycja nadgarstka poniżej łokcia - kontynuacja kierunku ramię-łokieć
                elbow_to_wrist_vector_x = neutral_wrist_offset_x
                elbow_to_wrist_vector_y = neutral_wrist_offset_y - int(idle_animation_amount)

                # Obliczamy pozycję nadgarstka
                right_wrist_x = right_elbow_pos[0] + elbow_to_wrist_vector_x
                right_wrist_y = right_elbow_pos[1] + elbow_to_wrist_vector_y

                # Aktualizacja pozycji nadgarstka z wygładzaniem
                if self.last_right_wrist is None:
                    self.last_right_wrist = (right_wrist_x, right_wrist_y)
                else:
                    new_x = int(
                        self.last_right_wrist[0] * self.smooth_factor + right_wrist_x * (1 - self.smooth_factor))
                    new_y = int(
                        self.last_right_wrist[1] * self.smooth_factor + right_wrist_y * (1 - self.smooth_factor))
                    self.last_right_wrist = (new_x, new_y)

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas aktualizacji pozycji rąk: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)}
            )

    def _draw_upper_body(self, canvas: np.ndarray) -> None:
        """
        Rysuje górną część ciała (popiersie) stick figure.

        Ta metoda koncentruje się wyłącznie na rysowaniu głowy, torsu i ramion,
        pomijając nogi i inne elementy dolnej części ciała.

        Args:
            canvas (np.ndarray): Płótno do rysowania
        """
        try:
            # 1. Głowa - okrąg
            cv2.circle(
                canvas,
                self.head_center,
                self.head_radius,
                self.figure_color,
                self.line_thickness
            )

            # 2. Obliczenie pozycji torsu i ramion względem głowy
            # Środek głowy
            head_x, head_y = self.head_center

            # Klatka piersiowa - poniżej głowy (krótsza niż w poprzedniej wersji)
            torso_top_y = head_y + self.head_radius + 5
            torso_length = int(self.head_radius * self.torso_length_factor)
            torso_bottom_y = torso_top_y + torso_length

            # 3. Rysowanie tułowia (linia pionowa od głowy w dół) - krótszy tułów
            cv2.line(
                canvas,
                (head_x, torso_top_y),
                (head_x, torso_bottom_y),
                self.figure_color,
                self.line_thickness
            )

            # 4. Rysowanie linii barków, jeśli mamy wykryte pozycje barków
            if self.last_left_shoulder and self.last_right_shoulder:
                cv2.line(
                    canvas,
                    self.last_left_shoulder,
                    self.last_right_shoulder,
                    self.figure_color,
                    self.line_thickness
                )
            else:
                # Domyślne rysowanie linii barków, jeśli nie mamy wykrytych pozycji
                shoulder_y = torso_top_y + int(self.head_radius * 0.3)
                shoulder_width = int(self.head_radius * self.shoulder_width_factor)
                left_shoulder_x = head_x - shoulder_width // 2
                right_shoulder_x = head_x + shoulder_width // 2

                cv2.line(
                    canvas,
                    (left_shoulder_x, shoulder_y),
                    (right_shoulder_x, shoulder_y),
                    self.figure_color,
                    self.line_thickness
                )

                # Aktualizacja domyślnych pozycji barków
                self.last_left_shoulder = (left_shoulder_x, shoulder_y)
                self.last_right_shoulder = (right_shoulder_x, shoulder_y)

            # 5. Rysowanie ramion (barki -> łokcie -> nadgarstki)
            # Lewe ramię - od barku do łokcia
            if self.last_left_shoulder and self.last_left_elbow:
                cv2.line(
                    canvas,
                    self.last_left_shoulder,
                    self.last_left_elbow,
                    self.figure_color,
                    self.line_thickness
                )

                # Lewe przedramię - od łokcia do nadgarstka
                if self.last_left_wrist:
                    cv2.line(
                        canvas,
                        self.last_left_elbow,
                        self.last_left_wrist,
                        self.figure_color,
                        self.line_thickness
                    )

            # Prawe ramię - od barku do łokcia
            if self.last_right_shoulder and self.last_right_elbow:
                cv2.line(
                    canvas,
                    self.last_right_shoulder,
                    self.last_right_elbow,
                    self.figure_color,
                    self.line_thickness
                )

                # Prawe przedramię - od łokcia do nadgarstka
                if self.last_right_wrist:
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
                f"Błąd podczas rysowania popiersia: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)}
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

        # Aktualizacja pozycji głowy na środku ekranu i wyżej (1/4 wysokości)
        self.head_center = (width // 2, height // 4)

        # Reset historii - bo zmieniamy skalę
        self.landmark_history = []
        self.last_left_shoulder = None
        self.last_right_shoulder = None
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
        self.last_left_shoulder = None
        self.last_right_shoulder = None
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
