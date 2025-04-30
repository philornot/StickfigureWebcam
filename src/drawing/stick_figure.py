#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/stick_figure.py

from typing import List, Tuple, Optional

import cv2
import numpy as np

from src.pose.pose_detector import PoseDetector
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class StickFigureRenderer:
    """
    Klasa do renderowania animowanej postaci stick figure na podstawie
    punktów charakterystycznych wykrytych przez PoseDetector.
    """

    def __init__(
            self,
            canvas_width: int = 640,
            canvas_height: int = 480,
            line_thickness: int = 3,
            head_radius_factor: float = 0.075,
            bg_color: Tuple[int, int, int] = (255, 255, 255),  # Białe tło
            figure_color: Tuple[int, int, int] = (0, 0, 0),  # Czarny ludzik
            chair_color: Tuple[int, int, int] = (150, 75, 0),  # Brązowe krzesło
            smooth_factor: float = 0.3,  # Współczynnik wygładzania ruchu
            smoothing_history: int = 3,  # Liczba klatek do wygładzania
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
            chair_color (Tuple[int, int, int]): Kolor krzesła (BGR)
            smooth_factor (float): Współczynnik wygładzania ruchu (0.0-1.0)
            smoothing_history (int): Liczba poprzednich klatek do wygładzania pozycji
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("StickFigureRenderer")

        # Parametry renderowania
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.line_thickness = line_thickness
        self.head_radius_factor = head_radius_factor
        self.bg_color = bg_color
        self.figure_color = figure_color
        self.chair_color = chair_color
        self.smooth_factor = smooth_factor

        # Obliczanie promienia głowy
        self.head_radius = int(head_radius_factor * canvas_height)

        # Bufor do wygładzania ruchu
        self.landmark_history = []
        self.smoothing_history = smoothing_history

        # Stałe - indeksy punktów z PoseDetector
        self.NOSE = PoseDetector.NOSE
        self.LEFT_EYE = PoseDetector.LEFT_EYE
        self.RIGHT_EYE = PoseDetector.RIGHT_EYE
        self.LEFT_SHOULDER = PoseDetector.LEFT_SHOULDER
        self.RIGHT_SHOULDER = PoseDetector.RIGHT_SHOULDER
        self.LEFT_ELBOW = PoseDetector.LEFT_ELBOW
        self.RIGHT_ELBOW = PoseDetector.RIGHT_ELBOW
        self.LEFT_WRIST = PoseDetector.LEFT_WRIST
        self.RIGHT_WRIST = PoseDetector.RIGHT_WRIST
        self.LEFT_HIP = PoseDetector.LEFT_HIP
        self.RIGHT_HIP = PoseDetector.RIGHT_HIP
        self.LEFT_KNEE = PoseDetector.LEFT_KNEE
        self.RIGHT_KNEE = PoseDetector.RIGHT_KNEE
        self.LEFT_ANKLE = PoseDetector.LEFT_ANKLE
        self.RIGHT_ANKLE = PoseDetector.RIGHT_ANKLE

        self.logger.info(
            "StickFigureRenderer",
            f"Renderer stick figure zainicjalizowany ({canvas_width}x{canvas_height})",
            log_type="DRAWING"
        )

    def render(
            self,
            landmarks: List[Tuple[float, float, float, float]],
            is_sitting: bool,
            confidence: float = 0.0
    ) -> np.ndarray:
        """
        Renderuje stick figure na podstawie punktów charakterystycznych.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów (x, y, z, visibility)
            is_sitting (bool): Czy osoba siedzi
            confidence (float): Pewność detekcji pozy (0.0-1.0)

        Returns:
            np.ndarray: Obraz z narysowanym stick figure
        """
        self.performance.start_timer()

        if landmarks is None or len(landmarks) < 33:  # MediaPipe Pose ma 33 punkty
            # Jeśli nie ma punktów, zwracamy pusty biały obraz
            self.logger.debug(
                "StickFigureRenderer",
                "Brak punktów do renderowania stick figure",
                log_type="DRAWING"
            )
            return np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

        try:
            # Wygładzanie punktów
            smooth_landmarks = self._smooth_landmarks(landmarks)

            # Tworzenie pustego obrazu (białe tło)
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas[:] = self.bg_color

            # Rysowanie stick figure w zależności od postawy
            if is_sitting:
                self._draw_sitting_figure(canvas, smooth_landmarks, confidence)
            else:
                self._draw_standing_figure(canvas, smooth_landmarks, confidence)

            self.performance.stop_timer()
            return canvas

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas renderowania stick figure: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)}
            )
            # W przypadku błędu zwracamy pusty biały obraz
            return np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

    def _smooth_landmarks(
            self,
            landmarks: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Wygładza ruch punktów charakterystycznych między klatkami.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            List[Tuple[float, float, float, float]]: Wygładzone punkty
        """
        # Dodaj bieżące punkty do historii
        self.landmark_history.append(landmarks)

        # Ogranicz rozmiar historii
        if len(self.landmark_history) > self.smoothing_history:
            self.landmark_history.pop(0)

        # Jeśli nie ma wystarczającej historii, zwróć bieżące punkty
        if len(self.landmark_history) <= 1:
            return landmarks

        # Wygładzanie punktów
        smoothed_landmarks = []

        # Dla każdego punktu
        for i in range(len(landmarks)):
            # Zbierz ten sam punkt z poprzednich klatek
            point_history = []
            for frame in self.landmark_history:
                if i < len(frame):
                    point_history.append(frame[i])

            # Jeśli nie ma historii dla tego punktu, użyj bieżącej wartości
            if not point_history:
                smoothed_landmarks.append(landmarks[i])
                continue

            # Oblicz wygładzoną wartość z większą wagą dla nowszych punktów
            x_sum, y_sum, z_sum, vis_sum = 0.0, 0.0, 0.0, 0.0
            weight_sum = 0.0

            for j, (x, y, z, vis) in enumerate(point_history):
                # Większa waga dla nowszych punktów
                weight = (j + 1) / len(point_history)
                weight_sum += weight

                x_sum += x * weight
                y_sum += y * weight
                z_sum += z * weight
                vis_sum += vis * weight

            # Normalizuj przez sumę wag
            if weight_sum > 0:
                x_smooth = x_sum / weight_sum
                y_smooth = y_sum / weight_sum
                z_smooth = z_sum / weight_sum
                vis_smooth = vis_sum / weight_sum

                # Zastosuj współczynnik wygładzania (im większy, tym bardziej wygładzony ruch)
                x_final = landmarks[i][0] * (1 - self.smooth_factor) + x_smooth * self.smooth_factor
                y_final = landmarks[i][1] * (1 - self.smooth_factor) + y_smooth * self.smooth_factor
                z_final = landmarks[i][2] * (1 - self.smooth_factor) + z_smooth * self.smooth_factor
                vis_final = landmarks[i][3] * (1 - self.smooth_factor) + vis_smooth * self.smooth_factor

                smoothed_landmarks.append((x_final, y_final, z_final, vis_final))
            else:
                smoothed_landmarks.append(landmarks[i])

        return smoothed_landmarks

    def _draw_standing_figure(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            confidence: float
    ) -> None:
        """
        Rysuje stick figure w pozycji stojącej.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            confidence (float): Pewność detekcji
        """
        h, w, _ = canvas.shape

        # 1. Rysowanie głowy
        self._draw_head(canvas, landmarks)

        # 2. Rysowanie tułowia
        self._draw_torso(canvas, landmarks)

        # 3. Rysowanie ramion i rąk
        self._draw_arms(canvas, landmarks)

        # 4. Rysowanie bioder i nóg
        self._draw_legs(canvas, landmarks)

        # 5. Dodanie opcjonalnych detali (np. kapelusz, uśmiech itp.) gdy wysoka pewność
        if confidence > 0.7:
            self._draw_details(canvas, landmarks, is_sitting=False)

    def _draw_sitting_figure(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            confidence: float
    ) -> None:
        """
        Rysuje stick figure w pozycji siedzącej wraz z krzesłem.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            confidence (float): Pewność detekcji
        """
        h, w, _ = canvas.shape

        # 1. Rysowanie krzesła
        self._draw_chair(canvas, landmarks)

        # 2. Rysowanie głowy
        self._draw_head(canvas, landmarks)

        # 3. Rysowanie tułowia
        self._draw_torso(canvas, landmarks)

        # 4. Rysowanie ramion i rąk
        self._draw_arms(canvas, landmarks)

        # 5. Rysowanie samych bioder (bez nóg lub z częściowo widocznymi udami)
        self._draw_sitting_legs(canvas, landmarks)  # Zmieniona nazwa dla większej precyzji

        # 6. Dodanie opcjonalnych detali
        if confidence > 0.7:
            self._draw_details(canvas, landmarks, is_sitting=True)

    def _draw_head(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje głowę stick figure.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape

        # Ustalamy pozycję głowy na podstawie nosa
        if landmarks[self.NOSE][3] > 0.5:  # Jeśli nos jest widoczny
            nose_x, nose_y = int(landmarks[self.NOSE][0] * w), int(landmarks[self.NOSE][1] * h)

            # Rysujemy okrąg głowy
            cv2.circle(canvas, (nose_x, nose_y), self.head_radius, self.figure_color, self.line_thickness)

            # Rysujemy oczy jeśli są widoczne
            eye_size = max(1, int(self.head_radius * 0.2))

            if landmarks[self.LEFT_EYE][3] > 0.5 and landmarks[self.RIGHT_EYE][3] > 0.5:
                left_eye_x = int(landmarks[self.LEFT_EYE][0] * w)
                left_eye_y = int(landmarks[self.LEFT_EYE][1] * h)
                right_eye_x = int(landmarks[self.RIGHT_EYE][0] * w)
                right_eye_y = int(landmarks[self.RIGHT_EYE][1] * h)

                # Alternatywnie: możemy użyć pozycji z nosa do obliczenia pozycji oczu
                # left_eye_x = nose_x - int(self.head_radius * 0.3)
                # left_eye_y = nose_y - int(self.head_radius * 0.2)
                # right_eye_x = nose_x + int(self.head_radius * 0.3)
                # right_eye_y = nose_y - int(self.head_radius * 0.2)

                cv2.circle(canvas, (left_eye_x, left_eye_y), eye_size, self.figure_color, -1)
                cv2.circle(canvas, (right_eye_x, right_eye_y), eye_size, self.figure_color, -1)

                # Rysowanie uśmiechu
                mouth_y = nose_y + int(self.head_radius * 0.3)
                cv2.ellipse(
                    canvas,
                    (nose_x, mouth_y),
                    (int(self.head_radius * 0.5), int(self.head_radius * 0.3)),
                    0, 0, 180, self.figure_color, self.line_thickness // 2
                )
        else:
            # Jeśli nos nie jest widoczny, używamy środka między barkami
            if landmarks[self.LEFT_SHOULDER][3] > 0.5 and landmarks[self.RIGHT_SHOULDER][3] > 0.5:
                left_shoulder_x = int(landmarks[self.LEFT_SHOULDER][0] * w)
                left_shoulder_y = int(landmarks[self.LEFT_SHOULDER][1] * h)
                right_shoulder_x = int(landmarks[self.RIGHT_SHOULDER][0] * w)
                right_shoulder_y = int(landmarks[self.RIGHT_SHOULDER][1] * h)

                center_x = (left_shoulder_x + right_shoulder_x) // 2
                center_y = min(left_shoulder_y, right_shoulder_y) - self.head_radius - 5

                # Rysujemy okrąg głowy
                cv2.circle(canvas, (center_x, center_y), self.head_radius, self.figure_color, self.line_thickness)

                # Proste oczy i uśmiech
                left_eye_x = center_x - int(self.head_radius * 0.3)
                left_eye_y = center_y - int(self.head_radius * 0.2)
                right_eye_x = center_x + int(self.head_radius * 0.3)
                right_eye_y = center_y - int(self.head_radius * 0.2)

                eye_size = max(1, int(self.head_radius * 0.15))
                cv2.circle(canvas, (left_eye_x, left_eye_y), eye_size, self.figure_color, -1)
                cv2.circle(canvas, (right_eye_x, right_eye_y), eye_size, self.figure_color, -1)

                # Rysowanie uśmiechu
                mouth_y = center_y + int(self.head_radius * 0.3)
                cv2.ellipse(
                    canvas,
                    (center_x, mouth_y),
                    (int(self.head_radius * 0.5), int(self.head_radius * 0.3)),
                    0, 0, 180, self.figure_color, self.line_thickness // 2
                )

    def _draw_torso(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje tułów stick figure.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape

        # Rysowanie linii między barkami
        if landmarks[self.LEFT_SHOULDER][3] > 0.5 and landmarks[self.RIGHT_SHOULDER][3] > 0.5:
            left_shoulder_x = int(landmarks[self.LEFT_SHOULDER][0] * w)
            left_shoulder_y = int(landmarks[self.LEFT_SHOULDER][1] * h)
            right_shoulder_x = int(landmarks[self.RIGHT_SHOULDER][0] * w)
            right_shoulder_y = int(landmarks[self.RIGHT_SHOULDER][1] * h)

            cv2.line(
                canvas,
                (left_shoulder_x, left_shoulder_y),
                (right_shoulder_x, right_shoulder_y),
                self.figure_color,
                self.line_thickness
            )

        # Rysowanie linii między biodrami
        if landmarks[self.LEFT_HIP][3] > 0.5 and landmarks[self.RIGHT_HIP][3] > 0.5:
            left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
            left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)
            right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
            right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)

            cv2.line(
                canvas,
                (left_hip_x, left_hip_y),
                (right_hip_x, right_hip_y),
                self.figure_color,
                self.line_thickness
            )

        # Rysowanie linii od lewego barku do lewego biodra
        if landmarks[self.LEFT_SHOULDER][3] > 0.5 and landmarks[self.LEFT_HIP][3] > 0.5:
            left_shoulder_x = int(landmarks[self.LEFT_SHOULDER][0] * w)
            left_shoulder_y = int(landmarks[self.LEFT_SHOULDER][1] * h)
            left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
            left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)

            cv2.line(
                canvas,
                (left_shoulder_x, left_shoulder_y),
                (left_hip_x, left_hip_y),
                self.figure_color,
                self.line_thickness
            )

        # Rysowanie linii od prawego barku do prawego biodra
        if landmarks[self.RIGHT_SHOULDER][3] > 0.5 and landmarks[self.RIGHT_HIP][3] > 0.5:
            right_shoulder_x = int(landmarks[self.RIGHT_SHOULDER][0] * w)
            right_shoulder_y = int(landmarks[self.RIGHT_SHOULDER][1] * h)
            right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
            right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)

            cv2.line(
                canvas,
                (right_shoulder_x, right_shoulder_y),
                (right_hip_x, right_hip_y),
                self.figure_color,
                self.line_thickness
            )

    def _draw_arms(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje ramiona i ręce stick figure.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape

        # Rysowanie lewej ręki (barki -> łokieć -> nadgarstek)
        if landmarks[self.LEFT_SHOULDER][3] > 0.5 and landmarks[self.LEFT_ELBOW][3] > 0.5:
            left_shoulder_x = int(landmarks[self.LEFT_SHOULDER][0] * w)
            left_shoulder_y = int(landmarks[self.LEFT_SHOULDER][1] * h)
            left_elbow_x = int(landmarks[self.LEFT_ELBOW][0] * w)
            left_elbow_y = int(landmarks[self.LEFT_ELBOW][1] * h)

            cv2.line(
                canvas,
                (left_shoulder_x, left_shoulder_y),
                (left_elbow_x, left_elbow_y),
                self.figure_color,
                self.line_thickness
            )

        if landmarks[self.LEFT_ELBOW][3] > 0.5 and landmarks[self.LEFT_WRIST][3] > 0.5:
            left_elbow_x = int(landmarks[self.LEFT_ELBOW][0] * w)
            left_elbow_y = int(landmarks[self.LEFT_ELBOW][1] * h)
            left_wrist_x = int(landmarks[self.LEFT_WRIST][0] * w)
            left_wrist_y = int(landmarks[self.LEFT_WRIST][1] * h)

            cv2.line(
                canvas,
                (left_elbow_x, left_elbow_y),
                (left_wrist_x, left_wrist_y),
                self.figure_color,
                self.line_thickness
            )

        # Rysowanie prawej ręki (barki -> łokieć -> nadgarstek)
        if landmarks[self.RIGHT_SHOULDER][3] > 0.5 and landmarks[self.RIGHT_ELBOW][3] > 0.5:
            right_shoulder_x = int(landmarks[self.RIGHT_SHOULDER][0] * w)
            right_shoulder_y = int(landmarks[self.RIGHT_SHOULDER][1] * h)
            right_elbow_x = int(landmarks[self.RIGHT_ELBOW][0] * w)
            right_elbow_y = int(landmarks[self.RIGHT_ELBOW][1] * h)

            cv2.line(
                canvas,
                (right_shoulder_x, right_shoulder_y),
                (right_elbow_x, right_elbow_y),
                self.figure_color,
                self.line_thickness
            )

        if landmarks[self.RIGHT_ELBOW][3] > 0.5 and landmarks[self.RIGHT_WRIST][3] > 0.5:
            right_elbow_x = int(landmarks[self.RIGHT_ELBOW][0] * w)
            right_elbow_y = int(landmarks[self.RIGHT_ELBOW][1] * h)
            right_wrist_x = int(landmarks[self.RIGHT_WRIST][0] * w)
            right_wrist_y = int(landmarks[self.RIGHT_WRIST][1] * h)

            cv2.line(
                canvas,
                (right_elbow_x, right_elbow_y),
                (right_wrist_x, right_wrist_y),
                self.figure_color,
                self.line_thickness
            )

        # Opcjonalnie: Rysowanie dłoni jako małych okręgów
        hand_radius = max(3, self.line_thickness)

        if landmarks[self.LEFT_WRIST][3] > 0.5:
            left_wrist_x = int(landmarks[self.LEFT_WRIST][0] * w)
            left_wrist_y = int(landmarks[self.LEFT_WRIST][1] * h)
            cv2.circle(canvas, (left_wrist_x, left_wrist_y), hand_radius, self.figure_color, -1)

        if landmarks[self.RIGHT_WRIST][3] > 0.5:
            right_wrist_x = int(landmarks[self.RIGHT_WRIST][0] * w)
            right_wrist_y = int(landmarks[self.RIGHT_WRIST][1] * h)
            cv2.circle(canvas, (right_wrist_x, right_wrist_y), hand_radius, self.figure_color, -1)

    def _draw_legs(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje nogi stick figure w pozycji stojącej.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape

        # Rysowanie lewej nogi (biodro -> kolano -> kostka)
        if landmarks[self.LEFT_HIP][3] > 0.5 and landmarks[self.LEFT_KNEE][3] > 0.5:
            left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
            left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)
            left_knee_x = int(landmarks[self.LEFT_KNEE][0] * w)
            left_knee_y = int(landmarks[self.LEFT_KNEE][1] * h)

            cv2.line(
                canvas,
                (left_hip_x, left_hip_y),
                (left_knee_x, left_knee_y),
                self.figure_color,
                self.line_thickness
            )

        if landmarks[self.LEFT_KNEE][3] > 0.5 and landmarks[self.LEFT_ANKLE][3] > 0.5:
            left_knee_x = int(landmarks[self.LEFT_KNEE][0] * w)
            left_knee_y = int(landmarks[self.LEFT_KNEE][1] * h)
            left_ankle_x = int(landmarks[self.LEFT_ANKLE][0] * w)
            left_ankle_y = int(landmarks[self.LEFT_ANKLE][1] * h)

            cv2.line(
                canvas,
                (left_knee_x, left_knee_y),
                (left_ankle_x, left_ankle_y),
                self.figure_color,
                self.line_thickness
            )

        # Rysowanie prawej nogi (biodro -> kolano -> kostka)
        if landmarks[self.RIGHT_HIP][3] > 0.5 and landmarks[self.RIGHT_KNEE][3] > 0.5:
            right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
            right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)
            right_knee_x = int(landmarks[self.RIGHT_KNEE][0] * w)
            right_knee_y = int(landmarks[self.RIGHT_KNEE][1] * h)

            cv2.line(
                canvas,
                (right_hip_x, right_hip_y),
                (right_knee_x, right_knee_y),
                self.figure_color,
                self.line_thickness
            )

        if landmarks[self.RIGHT_KNEE][3] > 0.5 and landmarks[self.RIGHT_ANKLE][3] > 0.5:
            right_knee_x = int(landmarks[self.RIGHT_KNEE][0] * w)
            right_knee_y = int(landmarks[self.RIGHT_KNEE][1] * h)
            right_ankle_x = int(landmarks[self.RIGHT_ANKLE][0] * w)
            right_ankle_y = int(landmarks[self.RIGHT_ANKLE][1] * h)

            cv2.line(
                canvas,
                (right_knee_x, right_knee_y),
                (right_ankle_x, right_ankle_y),
                self.figure_color,
                self.line_thickness
            )

        # Opcjonalnie: Rysowanie stóp jako małych linii
        foot_length = max(5, self.line_thickness * 3)

        if landmarks[self.LEFT_ANKLE][3] > 0.5:
            left_ankle_x = int(landmarks[self.LEFT_ANKLE][0] * w)
            left_ankle_y = int(landmarks[self.LEFT_ANKLE][1] * h)
            left_foot_x = left_ankle_x - foot_length

            cv2.line(
                canvas,
                (left_ankle_x, left_ankle_y),
                (left_foot_x, left_ankle_y),
                self.figure_color,
                self.line_thickness
            )

        if landmarks[self.RIGHT_ANKLE][3] > 0.5:
            right_ankle_x = int(landmarks[self.RIGHT_ANKLE][0] * w)
            right_ankle_y = int(landmarks[self.RIGHT_ANKLE][1] * h)
            right_foot_x = right_ankle_x + foot_length

            cv2.line(
                canvas,
                (right_ankle_x, right_ankle_y),
                (right_foot_x, right_ankle_y),
                self.figure_color,
                self.line_thickness
            )

    def _draw_sitting_legs(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje nogi stick figure w pozycji siedzącej.
        W tej pozycji nogi albo nie są rysowane wcale, albo rysowane są tylko
        górne części nóg, aby wyglądały jakby wychodziły z krzesła.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape

        # W pozycji siedzącej zwykle rysujemy tylko do kolan lub w ogóle nie rysujemy nóg
        # W tym przypadku dodajemy tylko opcjonalne górne części nóg

        if landmarks[self.LEFT_HIP][3] > 0.5 and landmarks[self.LEFT_KNEE][3] > 0.5:
            # Konwertuj współrzędne na piksele
            left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
            left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)
            left_knee_x = int(landmarks[self.LEFT_KNEE][0] * w)

            # Oblicz wirtualną pozycję kolana - wyprostowane przed siedziskiem
            # ale nie za długie - zwykle maksymalnie 1/3 wysokości od bioder
            max_leg_length = h / 6  # ograniczenie długości nogi
            virtual_knee_y = left_hip_y + int(max_leg_length)

            # Rysuj linię od biodra do wirtualnego kolana
            cv2.line(
                canvas,
                (left_hip_x, left_hip_y),
                (left_knee_x, virtual_knee_y),
                self.figure_color,
                self.line_thickness
            )

        if landmarks[self.RIGHT_HIP][3] > 0.5 and landmarks[self.RIGHT_KNEE][3] > 0.5:
            # Konwertuj współrzędne na piksele
            right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
            right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)
            right_knee_x = int(landmarks[self.RIGHT_KNEE][0] * w)

            # Oblicz wirtualną pozycję kolana - wyprostowane przed siedziskiem
            max_leg_length = h / 6  # ograniczenie długości nogi
            virtual_knee_y = right_hip_y + int(max_leg_length)

            # Rysuj linię od biodra do wirtualnego kolana
            cv2.line(
                canvas,
                (right_hip_x, right_hip_y),
                (right_knee_x, virtual_knee_y),
                self.figure_color,
                self.line_thickness
            )

    def _draw_chair(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje proste krzesło pod siedzącą postacią.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape

        # Określamy pozycję krzesła na podstawie bioder
        if landmarks[self.LEFT_HIP][3] > 0.5 and landmarks[self.RIGHT_HIP][3] > 0.5:
            left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
            left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)
            right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
            right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)

            # Środek bioder
            hip_center_x = (left_hip_x + right_hip_x) // 2
            hip_center_y = (left_hip_y + right_hip_y) // 2

            # Szerokość krzesła - bazująca na szerokości bioder z marginesem
            hip_width = abs(right_hip_x - left_hip_x)
            chair_width = int(hip_width * 1.5)
            chair_width = max(chair_width, 60)  # Minimalna szerokość

            # Parametry krzesła
            seat_height = 20  # Zwiększona grubość siedziska
            leg_height = int(h * 0.25)  # 25% wysokości obrazu

            # Pozycja siedziska krzesła - umieszczone centralnie pod biodrami
            seat_left = hip_center_x - chair_width // 2
            seat_right = hip_center_x + chair_width // 2
            seat_top = hip_center_y  # Ustawimy siedzisko dokładnie pod biodrami
            seat_bottom = seat_top + seat_height

            # Rysowanie siedziska jako wypełnionego prostokąta
            cv2.rectangle(
                canvas,
                (seat_left, seat_top),
                (seat_right, seat_bottom),
                self.chair_color,
                -1  # Wypełnione
            )

            # Krawędzie siedziska
            cv2.rectangle(
                canvas,
                (seat_left, seat_top),
                (seat_right, seat_bottom),
                (int(self.chair_color[0] * 0.7), int(self.chair_color[1] * 0.7), int(self.chair_color[2] * 0.7)),
                2  # Grubość krawędzi
            )

            # Rysowanie nóg krzesła
            leg_width = max(3, self.line_thickness)
            leg_margin = chair_width // 4

            # Lewa przednia noga
            cv2.rectangle(
                canvas,
                (seat_left + leg_margin, seat_bottom),
                (seat_left + leg_margin + leg_width, seat_bottom + leg_height),
                self.chair_color,
                -1
            )

            # Prawa przednia noga
            cv2.rectangle(
                canvas,
                (seat_right - leg_margin - leg_width, seat_bottom),
                (seat_right - leg_margin, seat_bottom + leg_height),
                self.chair_color,
                -1
            )

            # Lewa tylna noga
            cv2.rectangle(
                canvas,
                (seat_left + leg_margin // 2, seat_bottom),
                (seat_left + leg_margin // 2 + leg_width, seat_bottom + leg_height),
                (int(self.chair_color[0] * 0.8), int(self.chair_color[1] * 0.8), int(self.chair_color[2] * 0.8)),
                -1
            )

            # Prawa tylna noga
            cv2.rectangle(
                canvas,
                (seat_right - leg_margin // 2 - leg_width, seat_bottom),
                (seat_right - leg_margin // 2, seat_bottom + leg_height),
                (int(self.chair_color[0] * 0.8), int(self.chair_color[1] * 0.8), int(self.chair_color[2] * 0.8)),
                -1
            )

            # Oparcie krzesła
            backrest_height = int(h * 0.2)  # 20% wysokości obrazu
            backrest_width = chair_width // 10
            backrest_left = hip_center_x - backrest_width // 2

            # Narysuj oparcie jako pionowy prostokąt
            cv2.rectangle(
                canvas,
                (backrest_left, seat_top - backrest_height),
                (backrest_left + backrest_width, seat_top),
                self.chair_color,
                -1
            )

            # Górna część oparcia jako poziomy prostokąt
            top_width = chair_width // 2
            cv2.rectangle(
                canvas,
                (hip_center_x - top_width // 2, seat_top - backrest_height - 10),
                (hip_center_x + top_width // 2, seat_top - backrest_height),
                self.chair_color,
                -1
            )

    def _draw_details(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            is_sitting: bool
    ) -> None:
        """
        Rysuje dodatkowe detale stick figure (np. akcesoria, więcej detali twarzy).

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            is_sitting (bool): Czy osoba siedzi
        """
        h, w, _ = canvas.shape

        # Znajdujemy pozycję głowy
        head_pos = None

        if landmarks[self.NOSE][3] > 0.5:
            head_pos = (int(landmarks[self.NOSE][0] * w), int(landmarks[self.NOSE][1] * h))
        elif landmarks[self.LEFT_SHOULDER][3] > 0.5 and landmarks[self.RIGHT_SHOULDER][3] > 0.5:
            left_shoulder_x = int(landmarks[self.LEFT_SHOULDER][0] * w)
            left_shoulder_y = int(landmarks[self.LEFT_SHOULDER][1] * h)
            right_shoulder_x = int(landmarks[self.RIGHT_SHOULDER][0] * w)
            right_shoulder_y = int(landmarks[self.RIGHT_SHOULDER][1] * h)

            center_x = (left_shoulder_x + right_shoulder_x) // 2
            center_y = min(left_shoulder_y, right_shoulder_y) - self.head_radius - 5

            head_pos = (center_x, center_y)

        if head_pos:
            # Przykładowy detal: kapelusz/czapka
            hat_color = (100, 100, 100)  # Szary
            hat_width = int(self.head_radius * 2.2)
            hat_height = int(self.head_radius * 0.3)

            # Rysujemy kapelusz jako elipsę z wypełnieniem
            cv2.ellipse(
                canvas,
                (head_pos[0], head_pos[1] - int(self.head_radius * 0.5)),
                (hat_width // 2, hat_height),
                0, 0, 180, hat_color, -1
            )

            # Dodajemy rondo kapelusza
            cv2.ellipse(
                canvas,
                (head_pos[0], head_pos[1] - int(self.head_radius * 0.5)),
                (hat_width // 2 + 5, hat_height // 3),
                0, 180, 360, hat_color, -1
            )

    def set_colors(
            self,
            bg_color: Optional[Tuple[int, int, int]] = None,
            figure_color: Optional[Tuple[int, int, int]] = None,
            chair_color: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """
        Aktualizuje kolory używane do rysowania.

        Args:
            bg_color (Optional[Tuple[int, int, int]]): Nowy kolor tła (BGR)
            figure_color (Optional[Tuple[int, int, int]]): Nowy kolor stick figure (BGR)
            chair_color (Optional[Tuple[int, int, int]]): Nowy kolor krzesła (BGR)
        """
        if bg_color is not None:
            self.bg_color = bg_color

        if figure_color is not None:
            self.figure_color = figure_color

        if chair_color is not None:
            self.chair_color = chair_color

        self.logger.debug(
            "StickFigureRenderer",
            f"Zaktualizowano kolory (tło={self.bg_color}, figura={self.figure_color}, krzesło={self.chair_color})",
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
            f"Zaktualizowano grubość linii: {self.line_thickness}",
            log_type="DRAWING"
        )

    def set_smoothing(self, smooth_factor: float, history_length: int) -> None:
        """
        Aktualizuje parametry wygładzania ruchu.

        Args:
            smooth_factor (float): Nowy współczynnik wygładzania (0.0-1.0)
            history_length (int): Nowa długość historii
        """
        self.smooth_factor = max(0.0, min(1.0, smooth_factor))
        self.smoothing_history = max(1, history_length)

        # Dostosowanie historii do nowej długości
        if len(self.landmark_history) > self.smoothing_history:
            self.landmark_history = self.landmark_history[-self.smoothing_history:]

        self.logger.debug(
            "StickFigureRenderer",
            f"Zaktualizowano parametry wygładzania (factor={self.smooth_factor}, history={self.smoothing_history})",
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

        self.logger.info(
            "StickFigureRenderer",
            f"Zmieniono rozmiar płótna na {width}x{height}",
            log_type="DRAWING"
        )

        # Resetowanie historii wygładzania, ponieważ współrzędne będą w innej skali
        self.landmark_history = []

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan renderera.
        """
        self.landmark_history = []
        self.logger.debug("StickFigureRenderer", "Reset wewnętrznego stanu renderera", log_type="DRAWING")
