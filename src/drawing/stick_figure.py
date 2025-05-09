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
    Klasa do renderowania animowanej postaci stick figure (ludzika z kresek)
    na podstawie punktów charakterystycznych wykrytych przez PoseDetector.

    Implementuje prosty model stick figure składający się z:
    - Okrągłej głowy z podstawową mimiką
    - Pojedynczej kreski reprezentującej tułów
    - Dwóch kresek na każdą rękę (ramię + przedramię)
    - Dwóch kresek na każdą nogę (udo + podudzie)
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

            # Tworzenie pustego obrazu (białe tło)
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas[:] = self.bg_color

            # Rysowanie stick figure w zależności od postawy
            if is_sitting:
                self._draw_simplified_sitting_figure(canvas, smooth_landmarks, confidence)
            else:
                self._draw_simplified_standing_figure(canvas, smooth_landmarks, confidence)

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

    def _draw_simplified_standing_figure(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            confidence: float
    ) -> None:
        """
        Rysuje uproszczoną wersję stick figure w pozycji stojącej.

        Kluczowe zmiany:
        - Pojedyncza kreska dla tułowia
        - Dwie kreski dla każdej kończyny
        - Głowa przyczepiona do górnej części tułowia

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            confidence (float): Pewność detekcji
        """
        h, w, _ = canvas.shape

        # 1. Ustalenie punktu środkowego między barkami (górny koniec tułowia)
        if landmarks[self.LEFT_SHOULDER][3] > 0.5 and landmarks[self.RIGHT_SHOULDER][3] > 0.5:
            left_shoulder_x = int(landmarks[self.LEFT_SHOULDER][0] * w)
            left_shoulder_y = int(landmarks[self.LEFT_SHOULDER][1] * h)
            right_shoulder_x = int(landmarks[self.RIGHT_SHOULDER][0] * w)
            right_shoulder_y = int(landmarks[self.RIGHT_SHOULDER][1] * h)

            # Środek między barkami
            shoulder_center_x = (left_shoulder_x + right_shoulder_x) // 2
            shoulder_center_y = (left_shoulder_y + right_shoulder_y) // 2

            # 2. Ustalenie punktu środkowego między biodrami (dolny koniec tułowia)
            if landmarks[self.LEFT_HIP][3] > 0.5 and landmarks[self.RIGHT_HIP][3] > 0.5:
                left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
                left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)
                right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
                right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)

                # Środek między biodrami
                hip_center_x = (left_hip_x + right_hip_x) // 2
                hip_center_y = (left_hip_y + right_hip_y) // 2

                # 3. Rysowanie pojedynczej kreski tułowia
                cv2.line(
                    canvas,
                    (shoulder_center_x, shoulder_center_y),
                    (hip_center_x, hip_center_y),
                    self.figure_color,
                    self.line_thickness
                )

                # 4. Rysowanie nóg (dwie kreski na każdą nogę)
                self._draw_simplified_leg(
                    canvas,
                    landmarks,
                    self.LEFT_HIP,
                    self.LEFT_KNEE,
                    self.LEFT_ANKLE
                )

                self._draw_simplified_leg(
                    canvas,
                    landmarks,
                    self.RIGHT_HIP,
                    self.RIGHT_KNEE,
                    self.RIGHT_ANKLE
                )

            # 5. Rysowanie rąk (dwie kreski na każdą rękę)
            self._draw_simplified_arm(
                canvas,
                landmarks,
                self.LEFT_SHOULDER,
                self.LEFT_ELBOW,
                self.LEFT_WRIST
            )

            self._draw_simplified_arm(
                canvas,
                landmarks,
                self.RIGHT_SHOULDER,
                self.RIGHT_ELBOW,
                self.RIGHT_WRIST
            )

            # 6. Rysowanie głowy
            self._draw_simplified_head(canvas, landmarks, shoulder_center_x, shoulder_center_y)

    def _draw_simplified_sitting_figure(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            confidence: float
    ) -> None:
        """
        Rysuje uproszczoną wersję stick figure w pozycji siedzącej.

        Główne zmiany:
        - Krzesło rysowane pod środkiem bioder
        - Pojedyncza kreska dla tułowia
        - Skrócone lub niewidoczne nogi

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            confidence (float): Pewność detekcji
        """
        h, w, _ = canvas.shape

        # 1. Ustalenie punktu środkowego między barkami (górny koniec tułowia)
        if landmarks[self.LEFT_SHOULDER][3] > 0.5 and landmarks[self.RIGHT_SHOULDER][3] > 0.5:
            left_shoulder_x = int(landmarks[self.LEFT_SHOULDER][0] * w)
            left_shoulder_y = int(landmarks[self.LEFT_SHOULDER][1] * h)
            right_shoulder_x = int(landmarks[self.RIGHT_SHOULDER][0] * w)
            right_shoulder_y = int(landmarks[self.RIGHT_SHOULDER][1] * h)

            # Środek między barkami
            shoulder_center_x = (left_shoulder_x + right_shoulder_x) // 2
            shoulder_center_y = (left_shoulder_y + right_shoulder_y) // 2

            # 2. Ustalenie punktu środkowego między biodrami (dolny koniec tułowia)
            if landmarks[self.LEFT_HIP][3] > 0.5 and landmarks[self.RIGHT_HIP][3] > 0.5:
                left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
                left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)
                right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
                right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)

                # Środek między biodrami
                hip_center_x = (left_hip_x + right_hip_x) // 2
                hip_center_y = (left_hip_y + right_hip_y) // 2

                # 3. Najpierw rysujemy krzesło pod biodrami
                self._draw_simplified_chair(canvas, hip_center_x, hip_center_y)

                # 4. Rysowanie pojedynczej kreski tułowia
                cv2.line(
                    canvas,
                    (shoulder_center_x, shoulder_center_y),
                    (hip_center_x, hip_center_y),
                    self.figure_color,
                    self.line_thickness
                )

                # 5. Rysujemy krótkie nogi lub fragmenty nóg wystające z krzesła
                # Dla siedzącej postaci rysujemy jedynie górne części nóg (biodro-kolano)
                if landmarks[self.LEFT_HIP][3] > 0.5 and landmarks[self.LEFT_KNEE][3] > 0.5:
                    # Konwertujemy współrzędne
                    left_hip_x = int(landmarks[self.LEFT_HIP][0] * w)
                    left_hip_y = int(landmarks[self.LEFT_HIP][1] * h)
                    left_knee_x = int(landmarks[self.LEFT_KNEE][0] * w)

                    # Wirtualne kolano - krótsze niż rzeczywiste
                    knee_offset_y = int(self.head_radius * 1.5)
                    virtual_knee_y = left_hip_y + knee_offset_y

                    # Rysowanie górnej części nogi
                    cv2.line(
                        canvas,
                        (left_hip_x, left_hip_y),
                        (left_knee_x, virtual_knee_y),
                        self.figure_color,
                        self.line_thickness
                    )

                if landmarks[self.RIGHT_HIP][3] > 0.5 and landmarks[self.RIGHT_KNEE][3] > 0.5:
                    # Konwertujemy współrzędne
                    right_hip_x = int(landmarks[self.RIGHT_HIP][0] * w)
                    right_hip_y = int(landmarks[self.RIGHT_HIP][1] * h)
                    right_knee_x = int(landmarks[self.RIGHT_KNEE][0] * w)

                    # Wirtualne kolano - krótsze niż rzeczywiste
                    knee_offset_y = int(self.head_radius * 1.5)
                    virtual_knee_y = right_hip_y + knee_offset_y

                    # Rysowanie górnej części nogi
                    cv2.line(
                        canvas,
                        (right_hip_x, right_hip_y),
                        (right_knee_x, virtual_knee_y),
                        self.figure_color,
                        self.line_thickness
                    )

            # 6. Rysowanie rąk (dwie kreski na każdą rękę)
            self._draw_simplified_arm(
                canvas,
                landmarks,
                self.LEFT_SHOULDER,
                self.LEFT_ELBOW,
                self.LEFT_WRIST
            )

            self._draw_simplified_arm(
                canvas,
                landmarks,
                self.RIGHT_SHOULDER,
                self.RIGHT_ELBOW,
                self.RIGHT_WRIST
            )

            # 7. Rysowanie głowy
            self._draw_simplified_head(canvas, landmarks, shoulder_center_x, shoulder_center_y)

    def _draw_simplified_head(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            shoulder_center_x: int,
            shoulder_center_y: int
    ) -> None:
        """
        Rysuje uproszczoną głowę stick figure z prostą mimiką.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            shoulder_center_x (int): Współrzędna X środka między barkami
            shoulder_center_y (int): Współrzędna Y środka między barkami
        """
        h, w, _ = canvas.shape

        # Ustalamy pozycję głowy
        head_x, head_y = shoulder_center_x, shoulder_center_y - self.head_radius - 5

        # Jeśli nos jest widoczny, używamy jego pozycji
        if landmarks[self.NOSE][3] > 0.5:
            head_x = int(landmarks[self.NOSE][0] * w)
            head_y = int(landmarks[self.NOSE][1] * h)

        # Rysujemy okrąg głowy
        cv2.circle(canvas, (head_x, head_y), self.head_radius, self.figure_color, self.line_thickness)

        # Rysujemy oczy
        eye_size = max(2, int(self.head_radius * 0.15))
        eye_offset_x = int(self.head_radius * 0.3)
        eye_offset_y = int(self.head_radius * 0.1)

        left_eye_x = head_x - eye_offset_x
        right_eye_x = head_x + eye_offset_x

        eyes_y = head_y - eye_offset_y

        # Rysujemy czarne kropki jako oczy
        cv2.circle(canvas, (left_eye_x, eyes_y), eye_size, self.figure_color, -1)
        cv2.circle(canvas, (right_eye_x, eyes_y), eye_size, self.figure_color, -1)

        # Rysujemy uśmiech jako łuk
        smile_y = head_y + int(self.head_radius * 0.2)
        smile_width = int(self.head_radius * 0.6)
        smile_height = int(self.head_radius * 0.3)

        # W zależności od nastroju rysujemy odpowiednią minę
        if self.mood == "happy":
            # Uśmiech - krzywa w górę
            cv2.ellipse(
                canvas,
                (head_x, smile_y),
                (smile_width, smile_height),
                0, 0, 180, self.figure_color, self.line_thickness // 2
            )
        elif self.mood == "sad":
            # Smutek - krzywa w dół
            cv2.ellipse(
                canvas,
                (head_x, smile_y + smile_height * 2),
                (smile_width, smile_height),
                0, 180, 360, self.figure_color, self.line_thickness // 2
            )
        else:  # neutral
            # Neutralna linia prosta
            cv2.line(
                canvas,
                (head_x - smile_width, smile_y),
                (head_x + smile_width, smile_y),
                self.figure_color,
                self.line_thickness // 2
            )

    def _draw_simplified_arm(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            shoulder_idx: int,
            elbow_idx: int,
            wrist_idx: int
    ) -> None:
        """
        Rysuje uproszczoną rękę jako dwie linie: ramię i przedramię.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            shoulder_idx (int): Indeks punktu barku
            elbow_idx (int): Indeks punktu łokcia
            wrist_idx (int): Indeks punktu nadgarstka
        """
        h, w, _ = canvas.shape

        # Sprawdzamy widoczność punktów
        shoulder_visible = landmarks[shoulder_idx][3] > 0.5
        elbow_visible = landmarks[elbow_idx][3] > 0.5
        wrist_visible = landmarks[wrist_idx][3] > 0.5

        # Konwertujemy współrzędne
        if shoulder_visible:
            shoulder_x = int(landmarks[shoulder_idx][0] * w)
            shoulder_y = int(landmarks[shoulder_idx][1] * h)

            if elbow_visible:
                elbow_x = int(landmarks[elbow_idx][0] * w)
                elbow_y = int(landmarks[elbow_idx][1] * h)

                # Rysowanie ramienia (bark-łokieć)
                cv2.line(
                    canvas,
                    (shoulder_x, shoulder_y),
                    (elbow_x, elbow_y),
                    self.figure_color,
                    self.line_thickness
                )

                if wrist_visible:
                    wrist_x = int(landmarks[wrist_idx][0] * w)
                    wrist_y = int(landmarks[wrist_idx][1] * h)

                    # Rysowanie przedramienia (łokieć-nadgarstek)
                    cv2.line(
                        canvas,
                        (elbow_x, elbow_y),
                        (wrist_x, wrist_y),
                        self.figure_color,
                        self.line_thickness
                    )

                    # Opcjonalnie dodajemy małą kropkę jako dłoń
                    hand_radius = max(2, self.line_thickness // 2)
                    cv2.circle(canvas, (wrist_x, wrist_y), hand_radius, self.figure_color, -1)

    def _draw_simplified_leg(
            self,
            canvas: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            hip_idx: int,
            knee_idx: int,
            ankle_idx: int
    ) -> None:
        """
        Rysuje uproszczoną nogę jako dwie linie: udo i podudzie.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            hip_idx (int): Indeks punktu biodra
            knee_idx (int): Indeks punktu kolana
            ankle_idx (int): Indeks punktu kostki
        """
        h, w, _ = canvas.shape

        # Sprawdzamy widoczność punktów
        hip_visible = landmarks[hip_idx][3] > 0.5
        knee_visible = landmarks[knee_idx][3] > 0.5
        ankle_visible = landmarks[ankle_idx][3] > 0.5

        # Konwertujemy współrzędne
        if hip_visible:
            hip_x = int(landmarks[hip_idx][0] * w)
            hip_y = int(landmarks[hip_idx][1] * h)

            if knee_visible:
                knee_x = int(landmarks[knee_idx][0] * w)
                knee_y = int(landmarks[knee_idx][1] * h)

                # Rysowanie uda (biodro-kolano)
                cv2.line(
                    canvas,
                    (hip_x, hip_y),
                    (knee_x, knee_y),
                    self.figure_color,
                    self.line_thickness
                )

                if ankle_visible:
                    ankle_x = int(landmarks[ankle_idx][0] * w)
                    ankle_y = int(landmarks[ankle_idx][1] * h)

                    # Rysowanie podudzia (kolano-kostka)
                    cv2.line(
                        canvas,
                        (knee_x, knee_y),
                        (ankle_x, ankle_y),
                        self.figure_color,
                        self.line_thickness
                    )

                    # Opcjonalnie dodajemy małą kreskę jako stopę
                    foot_length = self.head_radius // 2
                    if hip_idx == self.LEFT_HIP:  # Lewa noga
                        foot_x = ankle_x - foot_length
                    else:  # Prawa noga
                        foot_x = ankle_x + foot_length

                    cv2.line(
                        canvas,
                        (ankle_x, ankle_y),
                        (foot_x, ankle_y),
                        self.figure_color,
                        self.line_thickness
                    )

    def _draw_simplified_chair(
            self,
            canvas: np.ndarray,
            hip_center_x: int,
            hip_center_y: int
    ) -> None:
        """
        Rysuje uproszczone krzesło pod siedzącą postacią.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            hip_center_x (int): Współrzędna X środka bioder
            hip_center_y (int): Współrzędna Y środka bioder
        """
        h, w, _ = canvas.shape

        # Ustalamy parametry krzesła
        chair_width = self.head_radius * 4
        seat_height = self.head_radius // 2
        leg_height = self.head_radius * 3

        # Pozycja siedziska
        seat_left = hip_center_x - chair_width // 2
        seat_right = hip_center_x + chair_width // 2
        seat_top = hip_center_y
        seat_bottom = seat_top + seat_height

        # Rysujemy siedzisko (prostokąt)
        cv2.rectangle(
            canvas,
            (seat_left, seat_top),
            (seat_right, seat_bottom),
            self.chair_color,
            -1  # Wypełnione
        )

        # Rysujemy krawędzie siedziska
        cv2.rectangle(
            canvas,
            (seat_left, seat_top),
            (seat_right, seat_bottom),
            (int(self.chair_color[0] * 0.7), int(self.chair_color[1] * 0.7), int(self.chair_color[2] * 0.7)),
            1  # Grubość krawędzi
        )

        # Szerokość nóg krzesła
        leg_width = max(2, self.line_thickness)

        # Rysujemy nogi krzesła (jako proste linie)
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

        # Rysujemy oparcie krzesła jako pionową linię
        backrest_height = self.head_radius * 2
        backrest_x = hip_center_x
        cv2.line(
            canvas,
            (backrest_x, seat_top),
            (backrest_x, seat_top - backrest_height),
            self.chair_color,
            leg_width
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

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan renderera.
        """
        self.landmark_history = []
        self.mood = "happy"  # Przywrócenie domyślnego nastroju
        self.logger.debug("StickFigureRenderer", "Reset wewnętrznego stanu renderera", log_type="DRAWING")
