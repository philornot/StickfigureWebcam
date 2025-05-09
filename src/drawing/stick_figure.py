#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/stick_figure.py

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from src.pose.pose_detector import PoseDetector
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class StickFigureRenderer:
    """
    Klasa do renderowania prostej postaci stick figure (ludzika z kresek)
    na podstawie punktów charakterystycznych wykrytych przez PoseDetector.

    Implementuje bardzo czytelny i uproszczony model stick figure:
    - Okrągła głowa z prostą mimiką (oczy i uśmiech)
    - Prosta linia reprezentująca tułów
    - Ręce i nogi jako proste linie bez zbędnych detali
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
        self.line_thickness = max(2, line_thickness)  # Minimum grubość 2 dla lepszej widoczności
        self.head_radius_factor = head_radius_factor
        self.bg_color = bg_color
        self.figure_color = figure_color
        self.chair_color = chair_color
        self.smooth_factor = smooth_factor

        # Obliczanie promienia głowy
        self.head_radius = int(head_radius_factor * canvas_height)

        # Bufor do wygładzania ruchu
        self.landmark_history: List[List[Tuple[float, float, float, float]]] = []
        self.smoothing_history = smoothing_history

        # Pamięć ostatnich poprawnych pozycji (do uzupełniania brakujących danych)
        self.last_valid_positions: Dict[int, Tuple[float, float, float, float]] = {}

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

        # Zmienna śledząca stan
        self.mood = "happy"  # Default mood: happy, sad, neutral

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

            # Tworzenie pustego obrazu
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas[:] = self.bg_color

            # Rysowanie stick figure w zależności od postawy
            if is_sitting:
                self._draw_sitting_figure(canvas, smooth_landmarks)
            else:
                self._draw_standing_figure(canvas, smooth_landmarks)

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
                weight = j + 1
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
        landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje stick figure w pozycji stojącej.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape
        points = {}

        # Konwersja punktów na współrzędne pixeli
        for i, (x, y, _, _) in enumerate(landmarks):
            points[i] = (int(x * w), int(y * h))

        # Rysowanie postaci
        try:
            # 1. Rysujemy tułów (linia od środka barków do środka bioder)
            shoulder_center = self._get_midpoint(points.get(self.LEFT_SHOULDER), points.get(self.RIGHT_SHOULDER))
            hip_center = self._get_midpoint(points.get(self.LEFT_HIP), points.get(self.RIGHT_HIP))

            if shoulder_center and hip_center:
                cv2.line(canvas, shoulder_center, hip_center, self.figure_color, self.line_thickness)

            # 2. Rysujemy ręce
            # Lewe ramię
            if self.LEFT_SHOULDER in points and self.LEFT_ELBOW in points:
                cv2.line(canvas, points[self.LEFT_SHOULDER], points[self.LEFT_ELBOW],
                         self.figure_color, self.line_thickness)

                # Lewe przedramię
                if self.LEFT_WRIST in points:
                    cv2.line(canvas, points[self.LEFT_ELBOW], points[self.LEFT_WRIST],
                             self.figure_color, self.line_thickness)

            # Prawe ramię
            if self.RIGHT_SHOULDER in points and self.RIGHT_ELBOW in points:
                cv2.line(canvas, points[self.RIGHT_SHOULDER], points[self.RIGHT_ELBOW],
                         self.figure_color, self.line_thickness)

                # Prawe przedramię
                if self.RIGHT_WRIST in points:
                    cv2.line(canvas, points[self.RIGHT_ELBOW], points[self.RIGHT_WRIST],
                             self.figure_color, self.line_thickness)

            # 3. Rysujemy nogi
            # Lewe udo
            if self.LEFT_HIP in points and self.LEFT_KNEE in points:
                cv2.line(canvas, points[self.LEFT_HIP], points[self.LEFT_KNEE],
                         self.figure_color, self.line_thickness)

                # Lewa goleń
                if self.LEFT_ANKLE in points:
                    cv2.line(canvas, points[self.LEFT_KNEE], points[self.LEFT_ANKLE],
                             self.figure_color, self.line_thickness)

            # Prawe udo
            if self.RIGHT_HIP in points and self.RIGHT_KNEE in points:
                cv2.line(canvas, points[self.RIGHT_HIP], points[self.RIGHT_KNEE],
                         self.figure_color, self.line_thickness)

                # Prawa goleń
                if self.RIGHT_ANKLE in points:
                    cv2.line(canvas, points[self.RIGHT_KNEE], points[self.RIGHT_ANKLE],
                             self.figure_color, self.line_thickness)

            # 4. Rysujemy głowę
            if self.NOSE in points:
                head_pos = points[self.NOSE]
                self._draw_simple_head(canvas, head_pos)
            elif shoulder_center:
                # Jeśli brak nosa, używamy pozycji nad barkami
                head_pos = (shoulder_center[0], shoulder_center[1] - self.head_radius - 10)
                self._draw_simple_head(canvas, head_pos)

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas rysowania postaci stojącej: {str(e)}",
                log_type="DRAWING"
            )

    def _draw_sitting_figure(
        self,
        canvas: np.ndarray,
        landmarks: List[Tuple[float, float, float, float]]
    ) -> None:
        """
        Rysuje stick figure w pozycji siedzącej wraz z krzesłem.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
        """
        h, w, _ = canvas.shape
        points = {}

        # Konwersja punktów na współrzędne pixeli
        for i, (x, y, _, _) in enumerate(landmarks):
            points[i] = (int(x * w), int(y * h))

        try:
            # 1. Ustalenie punktu środkowego bioder
            hip_center = None
            if self.LEFT_HIP in points and self.RIGHT_HIP in points:
                hip_center = self._get_midpoint(points[self.LEFT_HIP], points[self.RIGHT_HIP])
            elif self.LEFT_HIP in points:
                hip_center = points[self.LEFT_HIP]
            elif self.RIGHT_HIP in points:
                hip_center = points[self.RIGHT_HIP]

            # Jeśli nie mamy pozycji bioder, użyjmy pozycji barków
            shoulder_center = None
            if self.LEFT_SHOULDER in points and self.RIGHT_SHOULDER in points:
                shoulder_center = self._get_midpoint(points[self.LEFT_SHOULDER], points[self.RIGHT_SHOULDER])

                if not hip_center and shoulder_center:
                    # Oszacujmy pozycję bioder jako niżej od barków
                    hip_offset = int(self.head_radius * 3)
                    hip_center = (shoulder_center[0], shoulder_center[1] + hip_offset)

            # 2. Rysujemy krzesło
            if hip_center:
                self._draw_simple_chair(canvas, hip_center)

            # 3. Rysujemy tułów
            if shoulder_center and hip_center:
                cv2.line(canvas, shoulder_center, hip_center, self.figure_color, self.line_thickness)

            # 4. Rysujemy ręce
            # Lewe ramię
            if self.LEFT_SHOULDER in points and self.LEFT_ELBOW in points:
                cv2.line(canvas, points[self.LEFT_SHOULDER], points[self.LEFT_ELBOW],
                         self.figure_color, self.line_thickness)

                # Lewe przedramię
                if self.LEFT_WRIST in points:
                    cv2.line(canvas, points[self.LEFT_ELBOW], points[self.LEFT_WRIST],
                             self.figure_color, self.line_thickness)

            # Prawe ramię
            if self.RIGHT_SHOULDER in points and self.RIGHT_ELBOW in points:
                cv2.line(canvas, points[self.RIGHT_SHOULDER], points[self.RIGHT_ELBOW],
                         self.figure_color, self.line_thickness)

                # Prawe przedramię
                if self.RIGHT_WRIST in points:
                    cv2.line(canvas, points[self.RIGHT_ELBOW], points[self.RIGHT_WRIST],
                             self.figure_color, self.line_thickness)

            # 5. Skrócone nogi (tylko górne części)
            # Lewe udo - krótkie
            if self.LEFT_HIP in points and self.LEFT_KNEE in points:
                knee_pos = points[self.LEFT_KNEE]
                # Krótszy odcinek nogi
                hip_knee_vector = (knee_pos[0] - points[self.LEFT_HIP][0],
                                   knee_pos[1] - points[self.LEFT_HIP][1])
                shorter_knee_x = points[self.LEFT_HIP][0] + hip_knee_vector[0] // 2
                shorter_knee_y = points[self.LEFT_HIP][1] + hip_knee_vector[1] // 2
                shorter_knee = (shorter_knee_x, shorter_knee_y)

                cv2.line(canvas, points[self.LEFT_HIP], shorter_knee,
                         self.figure_color, self.line_thickness)

            # Prawe udo - krótkie
            if self.RIGHT_HIP in points and self.RIGHT_KNEE in points:
                knee_pos = points[self.RIGHT_KNEE]
                # Krótszy odcinek nogi
                hip_knee_vector = (knee_pos[0] - points[self.RIGHT_HIP][0],
                                   knee_pos[1] - points[self.RIGHT_HIP][1])
                shorter_knee_x = points[self.RIGHT_HIP][0] + hip_knee_vector[0] // 2
                shorter_knee_y = points[self.RIGHT_HIP][1] + hip_knee_vector[1] // 2
                shorter_knee = (shorter_knee_x, shorter_knee_y)

                cv2.line(canvas, points[self.RIGHT_HIP], shorter_knee,
                         self.figure_color, self.line_thickness)

            # 6. Rysujemy głowę
            if self.NOSE in points:
                head_pos = points[self.NOSE]
                self._draw_simple_head(canvas, head_pos)
            elif shoulder_center:
                # Jeśli brak nosa, używamy pozycji nad barkami
                head_pos = (shoulder_center[0], shoulder_center[1] - self.head_radius - 10)
                self._draw_simple_head(canvas, head_pos)

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas rysowania postaci siedzącej: {str(e)}",
                log_type="DRAWING"
            )

    def _draw_simple_head(self, canvas: np.ndarray, position: Tuple[int, int]) -> None:
        """
        Rysuje prostą głowę z podstawową mimiką.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            position (Tuple[int, int]): Pozycja środka głowy (x, y)
        """
        try:
            # Bezpieczne konwersje na int
            x, y = int(position[0]), int(position[1])

            # Rysujemy okrąg głowy
            cv2.circle(canvas, (x, y), self.head_radius, self.figure_color, self.line_thickness)

            # Oczy - proste kropki
            eye_offset_x = self.head_radius // 3
            eye_offset_y = self.head_radius // 4
            eye_size = max(2, self.head_radius // 8)

            cv2.circle(canvas, (x - eye_offset_x, y - eye_offset_y), eye_size, self.figure_color, -1)
            cv2.circle(canvas, (x + eye_offset_x, y - eye_offset_y), eye_size, self.figure_color, -1)

            # Usta - prosty uśmiech lub grymas
            mouth_y = y + self.head_radius // 3
            mouth_width = self.head_radius // 2

            if self.mood == "happy":
                # Uśmiech - prosty łuk
                cv2.ellipse(
                    canvas,
                    (x, mouth_y),
                    (mouth_width, mouth_width // 2),
                    0, 0, 180,  # 0-180 stopni = łuk w dół
                    self.figure_color,
                    max(1, self.line_thickness // 2)
                )
            elif self.mood == "sad":
                # Smutek - odwrócony łuk
                cv2.ellipse(
                    canvas,
                    (x, mouth_y + mouth_width // 2),
                    (mouth_width, mouth_width // 2),
                    0, 180, 360,  # 180-360 stopni = łuk w górę
                    self.figure_color,
                    max(1, self.line_thickness // 2)
                )
            else:
                # Neutralny - prosta linia
                cv2.line(
                    canvas,
                    (x - mouth_width, mouth_y),
                    (x + mouth_width, mouth_y),
                    self.figure_color,
                    max(1, self.line_thickness // 2)
                )

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas rysowania głowy: {str(e)}",
                log_type="DRAWING"
            )

    def _draw_simple_chair(self, canvas: np.ndarray, hip_position: Tuple[int, int]) -> None:
        """
        Rysuje proste krzesło pod pozycją bioder.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            hip_position (Tuple[int, int]): Pozycja bioder (x, y)
        """
        try:
            # Bezpieczne konwersje na int
            hip_x, hip_y = int(hip_position[0]), int(hip_position[1])

            # Parametry krzesła
            chair_width = self.head_radius * 4
            seat_height = self.head_radius // 2
            leg_height = self.head_radius * 2

            # Siedzisko - prostokąt
            seat_left = hip_x - chair_width // 2
            seat_right = hip_x + chair_width // 2
            seat_top = hip_y
            seat_bottom = hip_y + seat_height

            # Upewniamy się, że punkty są w granicach obrazu
            h, w, _ = canvas.shape
            seat_left = max(0, min(w - 1, seat_left))
            seat_right = max(0, min(w - 1, seat_right))
            seat_top = max(0, min(h - 1, seat_top))
            seat_bottom = max(0, min(h - 1, seat_bottom))

            # Rysujemy siedzisko
            cv2.rectangle(
                canvas,
                (seat_left, seat_top),
                (seat_right, seat_bottom),
                self.chair_color,
                -1  # wypełniony
            )

            # Nogi krzesła - dwie pionowe linie
            leg_width = max(1, self.line_thickness - 1)

            # Lewa noga
            left_leg_x = seat_left + chair_width // 4
            cv2.line(
                canvas,
                (left_leg_x, seat_bottom),
                (left_leg_x, seat_bottom + leg_height),
                self.chair_color,
                leg_width
            )

            # Prawa noga
            right_leg_x = seat_right - chair_width // 4
            cv2.line(
                canvas,
                (right_leg_x, seat_bottom),
                (right_leg_x, seat_bottom + leg_height),
                self.chair_color,
                leg_width
            )

            # Oparcie - prosta pionowa linia
            backrest_height = self.head_radius * 2
            cv2.line(
                canvas,
                (hip_x, seat_top),
                (hip_x, seat_top - backrest_height),
                self.chair_color,
                leg_width
            )

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Błąd podczas rysowania krzesła: {str(e)}",
                log_type="DRAWING"
            )

    def _get_midpoint(self, p1: Any, p2: Any) -> Optional[Tuple[int, int]]:
        """
        Bezpiecznie oblicza punkt środkowy między dwoma punktami.

        Args:
            p1: Pierwszy punkt (x, y)
            p2: Drugi punkt (x, y)

        Returns:
            Optional[Tuple[int, int]]: Punkt środkowy (x, y) lub None jeśli punkty niedostępne
        """
        if p1 is None or p2 is None:
            return None

        try:
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            return (int((x1 + x2) // 2), int((y1 + y2) // 2))
        except (TypeError, IndexError, ValueError):
            return None

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

    def set_mood(self, mood: str) -> None:
        """
        Ustawia nastrój stick figure, który wpływa na mimikę twarzy.

        Args:
            mood (str): Nastrój: "happy", "sad" lub "neutral"
        """
        valid_moods = ["happy", "sad", "neutral"]
        if mood in valid_moods:
            self.mood = mood
            self.logger.debug(
                "StickFigureRenderer",
                f"Ustawiono nastrój stick figure: {mood}",
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
        self.last_valid_positions = {}

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan renderera.
        """
        self.landmark_history = []
        self.last_valid_positions = {}
        self.mood = "happy"  # Przywrócenie domyślnego nastroju
        self.logger.debug("StickFigureRenderer", "Reset wewnętrznego stanu renderera", log_type="DRAWING")
