#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/face_renderer.py

from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np

from src.face.face_mesh_detector import FaceMeshDetector
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class FaceRenderer:
    """
    Klasa do renderowania ekspresyjnej twarzy dla stick figure na podstawie
    punktów charakterystycznych wykrytych przez FaceMeshDetector.

    Zapewnia bardziej szczegółowe i ekspresyjne rysowanie twarzy niż podstawowa
    funkcjonalność StickFigureRenderer, wykorzystując dane z MediaPipe FaceMesh.
    """

    def __init__(
        self,
        head_radius: int = 30,
        feature_color: Tuple[int, int, int] = (0, 0, 0),  # Czarny
        smooth_factor: float = 0.3,
        detail_level: str = "medium",  # "low", "medium", "high"
        logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja renderera twarzy.

        Args:
            head_radius (int): Promień głowy w pikselach
            feature_color (Tuple[int, int, int]): Kolor elementów twarzy (BGR)
            smooth_factor (float): Współczynnik wygładzania ruchu (0.0-1.0)
            detail_level (str): Poziom szczegółowości twarzy ("low", "medium", "high")
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("FaceRenderer")

        # Parametry renderowania
        self.head_radius = head_radius
        self.feature_color = feature_color
        self.smooth_factor = smooth_factor
        self.detail_level = detail_level

        # Domyślna skala elementów twarzy względem promienia głowy
        self.eye_scale = 0.2  # Oczy mają rozmiar ~20% promienia głowy
        self.mouth_scale = 0.5  # Usta mają szerokość ~50% promienia głowy

        # Bufor do wygładzania ruchu
        self.last_expressions = {
            "mouth_open": 0.0,
            "smile": 0.0,
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0
        }

        # Stałe - indeksy ważnych punktów FaceMesh
        self.FACE_LANDMARKS = {
            "nose_tip": FaceMeshDetector.NOSE_TIP,
            "left_eye": FaceMeshDetector.LEFT_EYE,
            "right_eye": FaceMeshDetector.RIGHT_EYE,
            "left_eyebrow": FaceMeshDetector.LEFT_EYEBROW,
            "right_eyebrow": FaceMeshDetector.RIGHT_EYEBROW,
            "lips_outer": FaceMeshDetector.LIPS_OUTER,
            "lips_inner": FaceMeshDetector.LIPS_INNER
        }

        # Dostosowanie szczegółowości
        self._adjust_detail_level(detail_level)

        self.logger.info(
            "FaceRenderer",
            f"Renderer twarzy zainicjalizowany (promień={head_radius}, szczegółowość={detail_level})",
            log_type="DRAWING"
        )

    def _adjust_detail_level(self, detail_level: str) -> None:
        """
        Dostosowuje parametry renderowania w zależności od wybranego poziomu szczegółowości.

        Args:
            detail_level (str): Poziom szczegółowości ("low", "medium", "high")
        """
        if detail_level == "low":
            # Prosty styl, podstawowe elementy
            self.draw_eyebrows = False
            self.draw_nose = False
            self.draw_detailed_eyes = False
            self.draw_detailed_mouth = False
        elif detail_level == "medium":
            # Średnio szczegółowy styl (domyślny)
            self.draw_eyebrows = True
            self.draw_nose = True
            self.draw_detailed_eyes = True
            self.draw_detailed_mouth = False
        elif detail_level == "high":
            # Bardzo szczegółowy styl
            self.draw_eyebrows = True
            self.draw_nose = True
            self.draw_detailed_eyes = True
            self.draw_detailed_mouth = True
        else:
            # Domyślnie średni poziom
            self.logger.warning(
                "FaceRenderer",
                f"Nieznany poziom szczegółowości: {detail_level}, używam 'medium'",
                log_type="DRAWING"
            )
            self.detail_level = "medium"
            self.draw_eyebrows = True
            self.draw_nose = True
            self.draw_detailed_eyes = True
            self.draw_detailed_mouth = False

        self.logger.debug(
            "FaceRenderer",
            f"Ustawiono poziom szczegółowości: {self.detail_level}",
            log_type="DRAWING"
        )

    def set_head_radius(self, radius: int) -> None:
        """
        Ustawia promień głowy.

        Args:
            radius (int): Nowy promień głowy w pikselach
        """
        self.head_radius = max(10, radius)  # Minimalny promień 10px
        self.logger.debug(
            "FaceRenderer",
            f"Zaktualizowano promień głowy: {self.head_radius}px",
            log_type="DRAWING"
        )

    def set_detail_level(self, detail_level: str) -> None:
        """
        Zmienia poziom szczegółowości renderowania twarzy.

        Args:
            detail_level (str): Poziom szczegółowości ("low", "medium", "high")
        """
        if detail_level in ["low", "medium", "high"]:
            self.detail_level = detail_level
            self._adjust_detail_level(detail_level)
            self.logger.info(
                "FaceRenderer",
                f"Zmieniono poziom szczegółowości na {detail_level}",
                log_type="DRAWING"
            )
        else:
            self.logger.warning(
                "FaceRenderer",
                f"Nieprawidłowy poziom szczegółowości: {detail_level}. Dozwolone wartości: low, medium, high",
                log_type="DRAWING"
            )

    def render_face(
        self,
        canvas: np.ndarray,
        head_center: Tuple[int, int],
        face_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Renderuje twarz na płótnie na podstawie danych z FaceMeshDetector.

        Args:
            canvas (np.ndarray): Płótno do rysowania (np.ndarray w formacie BGR)
            head_center (Tuple[int, int]): Współrzędne środka głowy (x, y) w pikselach
            face_data (Dict[str, Any]): Dane twarzy z FaceMeshDetector

        Returns:
            np.ndarray: Płótno z narysowaną twarzą
        """
        self.performance.start_timer()

        if not face_data or "has_face" not in face_data or not face_data["has_face"]:
            # Jeśli nie ma danych twarzy, rysujemy prostą uśmiechniętą buźkę
            self._draw_simple_face(canvas, head_center)
            self.performance.stop_timer()
            return canvas

        # Pobierz wyrazy twarzy z danych
        expressions = face_data.get("expressions", self.last_expressions)

        # Wygładzanie wyrazów twarzy między klatkami
        smoothed_expressions = self._smooth_expressions(expressions)

        # Zapisz wyrazy do użycia w przyszłych klatkach
        self.last_expressions = smoothed_expressions

        try:
            # Narysuj głowę (okrąg)
            cv2.circle(canvas, head_center, self.head_radius, self.feature_color, 2)

            # Narysuj elementy twarzy w zależności od poziomu szczegółowości
            if self.detail_level == "low":
                self._draw_simple_face_with_expressions(canvas, head_center, smoothed_expressions)
            else:
                self._draw_detailed_face(canvas, head_center, smoothed_expressions)

            self.performance.stop_timer()
            return canvas

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "FaceRenderer",
                f"Błąd podczas renderowania twarzy: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)}
            )
            # W przypadku błędu narysuj prostą buźkę
            self._draw_simple_face(canvas, head_center)
            return canvas

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
                smoothed[key] = (self.last_expressions[key] * self.smooth_factor +
                                 value * (1 - self.smooth_factor))
            else:
                smoothed[key] = value

        return smoothed

    def _draw_simple_face(self, canvas: np.ndarray, center: Tuple[int, int]) -> None:
        """
        Rysuje prostą buźkę bez wyrazów twarzy.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            center (Tuple[int, int]): Środek głowy (x, y)
        """
        # Środek głowy
        cx, cy = center

        # Rysuj oczy (dwa okręgi)
        eye_offset_x = int(self.head_radius * 0.3)
        eye_offset_y = int(self.head_radius * 0.2)
        eye_radius = max(2, int(self.head_radius * 0.15))

        # Lewe oko
        left_eye_pos = (cx - eye_offset_x, cy - eye_offset_y)
        cv2.circle(canvas, left_eye_pos, eye_radius, self.feature_color, -1)

        # Prawe oko
        right_eye_pos = (cx + eye_offset_x, cy - eye_offset_y)
        cv2.circle(canvas, right_eye_pos, eye_radius, self.feature_color, -1)

        # Rysuj uśmiech (prosty łuk)
        mouth_y = cy + int(self.head_radius * 0.2)
        mouth_width = int(self.head_radius * 0.6)
        mouth_height = int(self.head_radius * 0.3)

        # Rysujemy łuk - od 0 do 180 stopni
        cv2.ellipse(
            canvas,
            (cx, mouth_y),
            (mouth_width, mouth_height),
            0,  # kąt
            0, 180,  # start i koniec łuku
            self.feature_color,
            2  # grubość
        )

    def _draw_simple_face_with_expressions(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        expressions: Dict[str, float]
    ) -> None:
        """
        Rysuje prostą buźkę z uwzględnieniem wyrazów twarzy.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            center (Tuple[int, int]): Środek głowy (x, y)
            expressions (Dict[str, float]): Wartości wyrazów twarzy
        """
        # Środek głowy
        cx, cy = center

        # Rysuj oczy
        eye_offset_x = int(self.head_radius * 0.3)
        eye_offset_y = int(self.head_radius * 0.2)
        eye_radius = max(2, int(self.head_radius * 0.15))

        # Otwartość oczu
        left_eye_open = expressions.get("left_eye_open", 1.0)
        right_eye_open = expressions.get("right_eye_open", 1.0)

        # Lewe oko
        left_eye_pos = (cx - eye_offset_x, cy - eye_offset_y)
        if left_eye_open > 0.3:
            # Otwarte oko - okrąg
            cv2.circle(canvas, left_eye_pos, eye_radius, self.feature_color, -1)
        else:
            # Zamknięte oko - kreska
            cv2.line(
                canvas,
                (left_eye_pos[0] - eye_radius, left_eye_pos[1]),
                (left_eye_pos[0] + eye_radius, left_eye_pos[1]),
                self.feature_color,
                2
            )

        # Prawe oko
        right_eye_pos = (cx + eye_offset_x, cy - eye_offset_y)
        if right_eye_open > 0.3:
            # Otwarte oko - okrąg
            cv2.circle(canvas, right_eye_pos, eye_radius, self.feature_color, -1)
        else:
            # Zamknięte oko - kreska
            cv2.line(
                canvas,
                (right_eye_pos[0] - eye_radius, right_eye_pos[1]),
                (right_eye_pos[0] + eye_radius, right_eye_pos[1]),
                self.feature_color,
                2
            )

        # Rysuj usta z uwzględnieniem uśmiechu i otwarcia
        mouth_y = cy + int(self.head_radius * 0.2)
        mouth_width = int(self.head_radius * 0.6)
        mouth_height = int(self.head_radius * 0.3)

        # Wartości wyrazów twarzy
        mouth_open = expressions.get("mouth_open", 0.0)
        smile = expressions.get("smile", 0.0)

        if mouth_open > 0.2:
            # Otwarte usta - elipsa
            mouth_open_height = int(mouth_height * min(2.5, 1.0 + mouth_open * 2))
            cv2.ellipse(
                canvas,
                (cx, mouth_y),
                (mouth_width, mouth_open_height),
                0,  # kąt
                0, 360,  # pełna elipsa
                self.feature_color,
                2  # grubość
            )
        else:
            # Zamknięte usta - łuk (uśmiech)
            # Modyfikacja kształtu ust w zależności od uśmiechu
            start_angle = 0
            end_angle = 180

            if smile > 0.6:
                # Szeroki uśmiech
                mouth_height = int(mouth_height * 1.5)
            elif smile < 0.3:
                # Neutralne lub smutne usta
                start_angle = 180
                end_angle = 360
                mouth_y += mouth_height // 2

            cv2.ellipse(
                canvas,
                (cx, mouth_y),
                (mouth_width, mouth_height),
                0,  # kąt
                start_angle, end_angle,  # zakres łuku
                self.feature_color,
                2  # grubość
            )

    def _draw_detailed_face(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        expressions: Dict[str, float]
    ) -> None:
        """
        Rysuje szczegółową twarz z uwzględnieniem wyrazów twarzy.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            center (Tuple[int, int]): Środek głowy (x, y)
            expressions (Dict[str, float]): Wartości wyrazów twarzy
        """
        # Środek głowy
        cx, cy = center

        # 1. Rysowanie oczu
        eye_offset_x = int(self.head_radius * 0.3)
        eye_offset_y = int(self.head_radius * 0.2)
        eye_width = int(self.head_radius * 0.25)
        eye_height = int(self.head_radius * 0.15)

        # Wartości otwartości oczu
        left_eye_open = expressions.get("left_eye_open", 1.0)
        right_eye_open = expressions.get("right_eye_open", 1.0)

        # Lewe oko
        left_eye_center = (cx - eye_offset_x, cy - eye_offset_y)
        if self.draw_detailed_eyes:
            # Szczegółowe oko z regulowaną otwartością
            left_eye_h = int(max(1, eye_height * left_eye_open * 1.5))
            cv2.ellipse(
                canvas,
                left_eye_center,
                (eye_width, left_eye_h),
                0, 0, 360,  # pełna elipsa
                self.feature_color,
                1 if self.detail_level == "high" else 2  # cieńsza linia dla wysokiego poziomu szczegółowości
            )

            # Źrenica
            if left_eye_open > 0.3:
                pupil_size = max(2, int(eye_width * 0.6))
                cv2.circle(canvas, left_eye_center, pupil_size, self.feature_color, -1)
        else:
            # Proste oko
            if left_eye_open > 0.3:
                cv2.circle(canvas, left_eye_center, eye_width, self.feature_color, -1)
            else:
                cv2.line(
                    canvas,
                    (left_eye_center[0] - eye_width, left_eye_center[1]),
                    (left_eye_center[0] + eye_width, left_eye_center[1]),
                    self.feature_color,
                    2
                )

        # Prawe oko
        right_eye_center = (cx + eye_offset_x, cy - eye_offset_y)
        if self.draw_detailed_eyes:
            # Szczegółowe oko z regulowaną otwartością
            right_eye_h = int(max(1, eye_height * right_eye_open * 1.5))
            cv2.ellipse(
                canvas,
                right_eye_center,
                (eye_width, right_eye_h),
                0, 0, 360,  # pełna elipsa
                self.feature_color,
                1 if self.detail_level == "high" else 2
            )

            # Źrenica
            if right_eye_open > 0.3:
                pupil_size = max(2, int(eye_width * 0.6))
                cv2.circle(canvas, right_eye_center, pupil_size, self.feature_color, -1)
        else:
            # Proste oko
            if right_eye_open > 0.3:
                cv2.circle(canvas, right_eye_center, eye_width, self.feature_color, -1)
            else:
                cv2.line(
                    canvas,
                    (right_eye_center[0] - eye_width, right_eye_center[1]),
                    (right_eye_center[0] + eye_width, right_eye_center[1]),
                    self.feature_color,
                    2
                )

        # 2. Rysowanie brwi
        if self.draw_eyebrows:
            eyebrow_raised = expressions.get("eyebrow_raised", 0.0)
            eyebrow_offset_y = int(self.head_radius * (0.35 - eyebrow_raised * 0.15))
            eyebrow_width = int(self.head_radius * 0.28)

            # Lewy łuk brwi
            left_eyebrow_center = (cx - eye_offset_x, cy - eyebrow_offset_y)
            cv2.ellipse(
                canvas,
                left_eyebrow_center,
                (eyebrow_width, int(eyebrow_width * 0.4)),
                0,  # kąt
                180, 360,  # górny łuk
                self.feature_color,
                2
            )

            # Prawy łuk brwi
            right_eyebrow_center = (cx + eye_offset_x, cy - eyebrow_offset_y)
            cv2.ellipse(
                canvas,
                right_eyebrow_center,
                (eyebrow_width, int(eyebrow_width * 0.4)),
                0,  # kąt
                180, 360,  # górny łuk
                self.feature_color,
                2
            )

        # 3. Rysowanie nosa
        if self.draw_nose:
            nose_length = int(self.head_radius * 0.25)
            nose_top = (cx, cy - int(self.head_radius * 0.1))
            nose_bottom = (cx, cy + int(nose_length * 0.8))

            # Pionowa linia nosa
            cv2.line(canvas, nose_top, nose_bottom, self.feature_color, 2)

            # Dodajemy nozdrza dla wyższego poziomu szczegółowości
            if self.detail_level == "high":
                nostril_width = int(self.head_radius * 0.15)
                cv2.ellipse(
                    canvas,
                    nose_bottom,
                    (nostril_width, int(nostril_width * 0.4)),
                    0,  # kąt
                    180, 360,  # dolny łuk
                    self.feature_color,
                    1
                )

        # 4. Rysowanie ust
        mouth_y = cy + int(self.head_radius * 0.3)
        mouth_width = int(self.head_radius * 0.5)
        mouth_height = int(self.head_radius * 0.15)

        # Wartości ekspresji
        mouth_open = expressions.get("mouth_open", 0.0)
        smile = expressions.get("smile", 0.0)

        if self.draw_detailed_mouth:
            # Bardziej szczegółowe usta
            if mouth_open > 0.2:
                # Otwarte usta - dwie elipsy
                # Zewnętrzny kontur
                smile_offset = int(smile * mouth_height * 0.8)
                top_lip_y = mouth_y - smile_offset
                bottom_lip_y = mouth_y + int(mouth_open * mouth_height * 2) - smile_offset

                # Górna warga
                cv2.ellipse(
                    canvas,
                    (cx, top_lip_y),
                    (mouth_width, mouth_height),
                    0,  # kąt
                    0, 180,  # górny łuk
                    self.feature_color,
                    2
                )

                # Dolna warga
                cv2.ellipse(
                    canvas,
                    (cx, bottom_lip_y),
                    (mouth_width, mouth_height),
                    0,  # kąt
                    180, 360,  # dolny łuk
                    self.feature_color,
                    2
                )

                # Ciemne wnętrze ust
                inner_mouth_width = int(mouth_width * 0.8)
                inner_mouth_height = int(max(1, (bottom_lip_y - top_lip_y) * 0.7))
                inner_mouth_center = (cx, (top_lip_y + bottom_lip_y) // 2)

                # Wypełnione wnętrze ust
                cv2.ellipse(
                    canvas,
                    inner_mouth_center,
                    (inner_mouth_width, inner_mouth_height),
                    0, 0, 360,  # pełna elipsa
                    self.feature_color,
                    -1  # wypełnione
                )
            else:
                # Zamknięte usta
                # Główna linia ust
                mouth_curve = int(mouth_height * (smile - 0.5) * 2)

                if mouth_curve > 0:
                    # Uśmiech - łuk w dół
                    cv2.ellipse(
                        canvas,
                        (cx, mouth_y),
                        (mouth_width, abs(mouth_curve)),
                        0, 0, 180,
                        self.feature_color,
                        2
                    )
                else:
                    # Neutralne/smutne - łuk w górę
                    cv2.ellipse(
                        canvas,
                        (cx, mouth_y + abs(mouth_curve)),
                        (mouth_width, abs(mouth_curve)),
                        0, 180, 360,
                        self.feature_color,
                        2
                    )
        else:
            # Prostsze usta
            if mouth_open > 0.2:
                # Otwarte usta - elipsa
                mouth_open_height = int(mouth_height * min(3.0, 1.0 + mouth_open * 2.5))
                cv2.ellipse(
                    canvas,
                    (cx, mouth_y),
                    (mouth_width, mouth_open_height),
                    0, 0, 360,
                    self.feature_color,
                    2
                )

                # Ciemne wnętrze przy większym otwarciu
                if mouth_open > 0.4:
                    inner_width = int(mouth_width * 0.8)
                    inner_height = int(mouth_open_height * 0.8)
                    cv2.ellipse(
                        canvas,
                        (cx, mouth_y),
                        (inner_width, inner_height),
                        0, 0, 360,
                        self.feature_color,
                        -1
                    )
            else:
                # Zamknięte usta - łuk
                # Modyfikacja kształtu ust w zależności od uśmiechu
                start_angle = 0
                end_angle = 180

                if smile > 0.5:
                    # Uśmiech - łuk w dół
                    mouth_height = int(mouth_height * (0.8 + smile * 0.8))
                elif smile < 0.4:
                    # Smutek - łuk w górę
                    start_angle = 180
                    end_angle = 360
                    mouth_y += mouth_height // 2

                cv2.ellipse(
                    canvas,
                    (cx, mouth_y),
                    (mouth_width, mouth_height),
                    0, start_angle, end_angle,
                    self.feature_color,
                    2
                )

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan renderera.
        """
        # Resetuj bufor wygładzania
        self.last_expressions = {
            "mouth_open": 0.0,
            "smile": 0.0,
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0
        }

        self.logger.debug("FaceRenderer", "Reset wewnętrznego stanu renderera", log_type="DRAWING")
