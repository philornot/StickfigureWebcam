#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/face/face_mesh_detector.py

from typing import Dict, List, Tuple, Optional, Any

import cv2
import mediapipe as mp
import numpy as np

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class FaceMeshDetector:
    """
    Klasa do wykrywania twarzy i jej szczegółowych punktów charakterystycznych
    przy użyciu MediaPipe FaceMesh.

    Wykrywa 468 punktów charakterystycznych twarzy i dostarcza ich współrzędne
    dla renderowania bardziej szczegółowych i ekspresyjnych twarzy.
    """

    # Indeksy ważnych punktów twarzy
    # Krawędź twarzy (owal)
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    # Brwi
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    # Oczy
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    # Źrenice (przybliżone punkty)
    LEFT_PUPIL = 468  # W niektórych wersjach MediaPipe są dodatkowe punkty dla źrenic
    RIGHT_PUPIL = 473

    # Usta (zewnętrzna i wewnętrzna krawędź)
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

    # Dodatkowe punkty ust pomocne przy detekcji uśmiechu
    # Górna warga (środek)
    UPPER_LIP_CENTER = 13
    # Dolna warga (środek)
    LOWER_LIP_CENTER = 14
    # Kąciki ust
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    # Nos
    NOSE_TIP = 1
    NOSE_BOTTOM = 2
    NOSE_BRIDGE = [6, 197, 195, 5]

    # Policzki
    LEFT_CHEEK = 425
    RIGHT_CHEEK = 205

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja detektora twarzy FaceMesh.

        Args:
            max_num_faces (int): Maksymalna liczba twarzy do wykrywania (domyślnie 1)
            min_detection_confidence (float): Minimalna pewność detekcji (0.0-1.0)
            min_tracking_confidence (float): Minimalna pewność śledzenia (0.0-1.0)
            refine_landmarks (bool): Czy używać udoskonalonego modelu dla okolic oczu i ust
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("FaceMeshDetector")

        # Parametry detekcji
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks

        # Statystyki detekcji
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_score = 0.0
        self.last_face_data = None

        # Inicjalizacja MediaPipe FaceMesh
        self.logger.debug(
            "FaceMeshDetector",
            f"Inicjalizacja MediaPipe FaceMesh (max_faces={max_num_faces}, refine_landmarks={refine_landmarks})",
            log_type="FACE"
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # Inicjalizacja detektora twarzy
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Styl rysowania siatki twarzy
        self.face_mesh_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # BGR: zielony
            thickness=1,
            circle_radius=1
        )
        self.face_connections_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # BGR: niebieski
            thickness=1
        )

        self.logger.info(
            "FaceMeshDetector",
            "MediaPipe FaceMesh zainicjalizowany pomyślnie",
            log_type="FACE"
        )

    def detect_face(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Wykrywa twarz i punkty charakterystyczne na obrazie.

        Args:
            image (np.ndarray): Obraz wejściowy w formacie BGR (OpenCV)

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - bool: True jeśli wykryto twarz, False w przeciwnym razie
                - Dict: Słownik zawierający informacje o wykrytej twarzy
        """
        self.performance.start_timer()
        self.frame_count += 1

        # MediaPipe wymaga obrazu w formacie RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Detekcja twarzy
        try:
            results = self.face_mesh.process(image_rgb)

            # Przygotowanie wyników
            face_data = {
                "landmarks": None,
                "multi_face_landmarks": None,
                "detection_score": 0.0,
                "has_face": False,
                "frame_height": h,
                "frame_width": w,
                "expressions": {
                    "mouth_open": 0.0,
                    "smile": 0.0,
                    "left_eye_open": 1.0,
                    "right_eye_open": 1.0,
                    "eyebrow_raised": 0.0,
                    "surprise": 0.0
                },
                "debug_info": {}  # Dodane pole do debugowania
            }

            # Sprawdzenie czy twarz została wykryta
            if results.multi_face_landmarks:
                # Twarz wykryta
                self.detection_count += 1
                face_data["has_face"] = True
                face_data["multi_face_landmarks"] = results.multi_face_landmarks

                # Jeśli wykryto wiele twarzy, bierzemy tylko pierwszą
                face_landmarks = results.multi_face_landmarks[0]

                # Konwersja punktów charakterystycznych na listę krotek (x, y, z, visibility)
                # Uwaga: MediaPipe FaceMesh nie dostarcza visibility, używamy wartości 1.0
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z, 1.0))

                face_data["landmarks"] = landmarks

                # Analizujemy wyraz twarzy
                face_data["expressions"] = self._analyze_expressions(landmarks, w, h)

                # Szacowanie pewności detekcji - MediaPipe FaceMesh nie zwraca score,
                # więc używamy stałej wartości dla wykrytej twarzy
                face_data["detection_score"] = 1.0
                self.last_detection_score = 1.0
                self.last_face_data = face_data

                # Logowanie co 30 klatek
                if self.frame_count % 30 == 0:
                    self.logger.debug(
                        "FaceMeshDetector",
                        f"Wykryto twarz, ekspresje: {face_data['expressions']}",
                        log_type="FACE"
                    )

                # Monitorowanie wydajności
                self.performance.stop_timer()
                execution_time = self.performance.get_last_execution_time() * 1000  # ms

                # Statystyki wydajności co 100 detekcji
                if self.detection_count % 100 == 0:
                    self.logger.info(
                        "FaceMeshDetector",
                        f"Wykryto {self.detection_count} twarzy, "
                        f"stosunek detekcji: {self.detection_count / self.frame_count:.2f}",
                        log_type="FACE"
                    )
                    self.logger.performance_metrics(0, execution_time, "FaceMeshDetector")

                return True, face_data
            else:
                # Brak detekcji twarzy
                self.performance.stop_timer()

                # Jeśli mamy ostatnią wykrytą twarz, używamy jej z obniżoną pewnością
                if self.last_face_data is not None:
                    # Kopiujemy ostatnie dane, ale obniżamy pewność
                    face_data = self.last_face_data.copy()
                    face_data["detection_score"] *= 0.8  # Obniżamy pewność
                    face_data["has_face"] = face_data["detection_score"] > 0.5

                    # Logowanie co 50 klatek bez świeżej detekcji
                    if self.frame_count % 50 == 0:
                        self.logger.debug(
                            "FaceMeshDetector",
                            "Używam ostatniej wykrytej twarzy z obniżoną pewnością",
                            log_type="FACE"
                        )

                    return face_data["has_face"], face_data

                # Logowanie braku detekcji co 50 klatek bez detekcji
                if self.frame_count % 50 == 0 and self.detection_count == 0:
                    self.logger.warning(
                        "FaceMeshDetector",
                        "Nie wykryto twarzy w ostatnich 50 klatkach",
                        log_type="FACE"
                    )
                elif self.frame_count % 100 == 0:
                    self.logger.debug(
                        "FaceMeshDetector",
                        f"Brak detekcji twarzy, stosunek detekcji: {self.detection_count / self.frame_count:.2f}",
                        log_type="FACE"
                    )

                return False, face_data

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "FaceMeshDetector",
                f"Błąd podczas detekcji twarzy: {str(e)}",
                log_type="FACE",
                error={"error": str(e)}
            )
            return False, {"error": str(e), "has_face": False}

    def _analyze_expressions(
        self,
        landmarks: List[Tuple[float, float, float, float]],
        img_width: int,
        img_height: int
    ) -> Dict[str, float]:
        """
        Analizuje wyraz twarzy na podstawie punktów charakterystycznych.

        Ulepszono detekcję uśmiechu przez analizę kształtu ust i położenia kącików.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów charakterystycznych
            img_width (int): Szerokość obrazu
            img_height (int): Wysokość obrazu

        Returns:
            Dict[str, float]: Słownik z wartościami ekspresji twarzy (0.0-1.0)
        """
        expressions = {
            "mouth_open": 0.0,
            "smile": 0.0,
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0
        }

        # Słownik z wartościami debug dla analizy
        debug_values = {}

        try:
            if not landmarks or len(landmarks) < 468:  # FaceMesh ma 468 punktów
                return expressions

            # Analiza otwarcia ust
            # Używamy punktów górnej i dolnej wargi do określenia otwarcia ust
            top_lip = landmarks[self.UPPER_LIP_CENTER]  # Górna warga, środek
            bottom_lip = landmarks[self.LOWER_LIP_CENTER]  # Dolna warga, środek

            # Obliczamy odległość między wargami w pikselach
            lips_distance = abs(top_lip[1] - bottom_lip[1]) * img_height

            # Normalizacja względem rozmiaru twarzy
            face_height = abs(landmarks[10][1] - landmarks[152][1]) * img_height
            if face_height > 0:
                # Przeliczamy na wartość 0.0-1.0, gdzie 0.0 to zamknięte usta, 1.0 to szeroko otwarte
                mouth_open_ratio = min(1.0, (lips_distance / face_height) * 5.0)
                expressions["mouth_open"] = mouth_open_ratio
                debug_values["lips_distance"] = lips_distance
                debug_values["face_height"] = face_height
                debug_values["mouth_open_ratio_raw"] = lips_distance / face_height

            # ===== ULEPSZONY ALGORYTM ANALIZY UŚMIECHU =====
            # 1. Obliczamy różnicę wysokości między kącikami ust a środkiem górnej wargi
            left_mouth_corner = landmarks[self.LEFT_MOUTH_CORNER]  # Lewy kącik ust
            right_mouth_corner = landmarks[self.RIGHT_MOUTH_CORNER]  # Prawy kącik ust
            center_upper_lip = landmarks[self.UPPER_LIP_CENTER]  # Środek górnej wargi

            # Przechowujemy współrzędne Y (gdzie mniejsze Y jest wyżej na ekranie)
            left_corner_y = left_mouth_corner[1]
            right_corner_y = right_mouth_corner[1]
            center_upper_y = center_upper_lip[1]

            # 2. Kluczowa modyfikacja: Obliczamy krzywość ust
            # Obliczamy o ile kąciki ust są wyżej od środka górnej wargi
            # Przy uśmiechu kąciki będą wyżej (mniejsze Y) niż środek
            left_up_diff = center_upper_y - left_corner_y
            right_up_diff = center_upper_y - right_corner_y

            # Uśmiech jest symetryczny, więc oba kąciki powinny być uniesione
            corner_diff = (left_up_diff + right_up_diff) / 2.0

            # Wartości dodatnie oznaczają, że kąciki są wyżej niż środek - to uśmiech
            # Normalizacja i ograniczenie do zakresu 0.0-1.0
            # Mnożymy przez 4.0 zamiast 2.0 dla zwiększenia czułości
            smile_ratio = max(0.0, min(1.0, corner_diff * 4.0))

            # 3. Dodatkowy czynnik: badamy szerokość ust
            # Przy uśmiechu usta są szersze
            mouth_width = abs(left_mouth_corner[0] - right_mouth_corner[0]) * img_width

            # Normalizujemy względem szerokości twarzy
            face_width = abs(landmarks[234][0] - landmarks[454][0]) * img_width
            if face_width > 0:
                width_ratio = min(1.0, (mouth_width / face_width) * 2.0)
                # Uwzględniamy szerokość jako dodatkowy czynnik (z mniejszą wagą)
                smile_ratio = smile_ratio * 0.8 + width_ratio * 0.2

            # Zapisujemy ostateczną wartość
            expressions["smile"] = smile_ratio

            # Zapisujemy wartości debug do analizy
            debug_values["left_corner_y"] = left_corner_y
            debug_values["right_corner_y"] = right_corner_y
            debug_values["center_upper_y"] = center_upper_y
            debug_values["corner_diff"] = corner_diff
            debug_values["mouth_width"] = mouth_width
            debug_values["face_width"] = face_width
            debug_values["width_ratio"] = width_ratio if 'width_ratio' in locals() else None
            debug_values["smile_ratio_final"] = smile_ratio

            # Analiza otwarcia oczu
            # Używamy punktów górnej i dolnej powieki do określenia otwarcia oka
            # Lewe oko
            left_eye_top = landmarks[159]  # Górna powieka lewego oka
            left_eye_bottom = landmarks[145]  # Dolna powieka lewego oka
            left_eye_distance = abs(left_eye_top[1] - left_eye_bottom[1]) * img_height

            # Prawe oko
            right_eye_top = landmarks[386]  # Górna powieka prawego oka
            right_eye_bottom = landmarks[374]  # Dolna powieka prawego oka
            right_eye_distance = abs(right_eye_top[1] - right_eye_bottom[1]) * img_height

            # Normalizacja względem rozmiaru twarzy
            if face_height > 0:
                eye_open_norm = 8.0  # Współczynnik normalizacji
                left_eye_open_ratio = min(1.0, (left_eye_distance / face_height) * eye_open_norm)
                right_eye_open_ratio = min(1.0, (right_eye_distance / face_height) * eye_open_norm)

                # Wartości bliskie 0 oznaczają zamknięte oczy
                expressions["left_eye_open"] = left_eye_open_ratio
                expressions["right_eye_open"] = right_eye_open_ratio

            # Analiza uniesienia brwi
            # Używamy punktów brwi i oczu do określenia uniesienia brwi
            left_eyebrow = landmarks[107]  # Punkt na lewej brwi
            left_eye_position = landmarks[159]  # Punkt na lewym oku

            right_eyebrow = landmarks[336]  # Punkt na prawej brwi
            right_eye_position = landmarks[386]  # Punkt na prawym oku

            # Obliczamy odległość między brwiami a oczami
            left_eyebrow_distance = abs(left_eyebrow[1] - left_eye_position[1]) * img_height
            right_eyebrow_distance = abs(right_eyebrow[1] - right_eye_position[1]) * img_height
            avg_eyebrow_distance = (left_eyebrow_distance + right_eyebrow_distance) / 2

            # Normalizacja względem rozmiaru twarzy
            if face_height > 0:
                eyebrow_raised_ratio = min(1.0, (avg_eyebrow_distance / face_height) * 10.0)
                expressions["eyebrow_raised"] = eyebrow_raised_ratio

            # Analiza zaskoczenia (kombinacja otwarcia oczu, ust i uniesienia brwi)
            surprise_score = (expressions["mouth_open"] + expressions["eyebrow_raised"] +
                              (expressions["left_eye_open"] + expressions["right_eye_open"]) / 2) / 3
            expressions["surprise"] = min(1.0, surprise_score)

            # Dodajemy debugowanie - co 100 klatek logujemy wartości uśmiechu
            if self.frame_count % 100 == 0:
                self.logger.debug(
                    "FaceMeshDetector",
                    f"Analiza uśmiechu: wartość={expressions['smile']:.2f}, "
                    f"diff_lewy={left_up_diff:.4f}, diff_prawy={right_up_diff:.4f}",
                    log_type="FACE"
                )

        except Exception as e:
            self.logger.error(
                "FaceMeshDetector",
                f"Błąd podczas analizy wyrazu twarzy: {str(e)}",
                log_type="FACE"
            )

        return expressions

    def draw_face_mesh(
        self,
        image: np.ndarray,
        face_landmarks: Any,
        draw_tesselation: bool = True,
        draw_contours: bool = True,
        draw_irises: bool = True
    ) -> np.ndarray:
        """
        Rysuje siatkę twarzy na obrazie.

        Args:
            image (np.ndarray): Obraz wejściowy (BGR)
            face_landmarks: Obiekt multi_face_landmarks z MediaPipe
            draw_tesselation (bool): Czy rysować pełną siatkę twarzy
            draw_contours (bool): Czy rysować kontury twarzy (oczy, usta, brwi)
            draw_irises (bool): Czy rysować tęczówki (jeśli są dostępne)

        Returns:
            np.ndarray: Obraz z narysowaną siatką twarzy
        """
        if face_landmarks is None:
            return image

        # Tworzymy kopię obrazu
        img_copy = image.copy()

        # Dla każdej wykrytej twarzy
        for face_landmark in face_landmarks:
            # Rysujemy pełną siatkę twarzy
            if draw_tesselation:
                self.mp_drawing.draw_landmarks(
                    image=img_copy,
                    landmark_list=face_landmark,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # Rysujemy kontury (oczy, usta, brwi, owal twarzy)
            if draw_contours:
                self.mp_drawing.draw_landmarks(
                    image=img_copy,
                    landmark_list=face_landmark,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            # Rysujemy tęczówki (jeśli są dostępne w używanej wersji MediaPipe)
            if draw_irises and hasattr(self.mp_face_mesh, 'FACEMESH_IRISES'):
                self.mp_drawing.draw_landmarks(
                    image=img_copy,
                    landmark_list=face_landmark,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        return img_copy

    def draw_simplified_face(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float, float, float]],
        expressions: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Rysuje uproszczoną twarz z ważnymi elementami.

        Args:
            image (np.ndarray): Obraz wejściowy (BGR)
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            expressions (Dict[str, float], optional): Słownik z wartościami ekspresji twarzy

        Returns:
            np.ndarray: Obraz z narysowaną uproszczoną twarzą
        """
        if landmarks is None or len(landmarks) < 468:
            return image

        # Tworzymy kopię obrazu
        img_copy = image.copy()
        h, w, _ = img_copy.shape

        # Jeśli nie podano ekspresji, używamy pustych
        if expressions is None:
            expressions = {
                "mouth_open": 0.0,
                "smile": 0.0,
                "left_eye_open": 1.0,
                "right_eye_open": 1.0,
                "eyebrow_raised": 0.0,
                "surprise": 0.0
            }

        try:
            # Rysowanie owalu twarzy
            face_points = []
            for idx in self.FACE_OVAL:
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                face_points.append((x, y))

            if face_points:
                cv2.polylines(img_copy, [np.array(face_points)], True, (0, 255, 0), 2)

            # Rysowanie oczu
            left_eye_points = []
            for idx in self.LEFT_EYE:
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                left_eye_points.append((x, y))

            right_eye_points = []
            for idx in self.RIGHT_EYE:
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                right_eye_points.append((x, y))

            if left_eye_points:
                # Otwartość oka wpływa na to, jak jest rysowane
                if expressions["left_eye_open"] < 0.2:
                    # Rysuj zamknięte oko (linia)
                    left_eye_center = np.mean(np.array(left_eye_points), axis=0).astype(int)
                    left_eye_width = int(np.max(np.array(left_eye_points)[:, 0]) -
                                         np.min(np.array(left_eye_points)[:, 0]))
                    cv2.line(img_copy,
                             (left_eye_center[0] - left_eye_width // 2, left_eye_center[1]),
                             (left_eye_center[0] + left_eye_width // 2, left_eye_center[1]),
                             (0, 0, 255), 2)
                else:
                    # Rysuj otwarte oko (kontur)
                    cv2.polylines(img_copy, [np.array(left_eye_points)], True, (0, 0, 255), 2)

                    # Rysuj źrenicę
                    left_eye_center = np.mean(np.array(left_eye_points), axis=0).astype(int)
                    pupil_radius = max(1, int((np.max(np.array(left_eye_points)[:, 0]) -
                                               np.min(np.array(left_eye_points)[:, 0])) * 0.15))
                    cv2.circle(img_copy, tuple(left_eye_center), pupil_radius, (0, 0, 0), -1)

            if right_eye_points:
                # Otwartość oka wpływa na to, jak jest rysowane
                if expressions["right_eye_open"] < 0.2:
                    # Rysuj zamknięte oko (linia)
                    right_eye_center = np.mean(np.array(right_eye_points), axis=0).astype(int)
                    right_eye_width = int(np.max(np.array(right_eye_points)[:, 0]) -
                                          np.min(np.array(right_eye_points)[:, 0]))
                    cv2.line(img_copy,
                             (right_eye_center[0] - right_eye_width // 2, right_eye_center[1]),
                             (right_eye_center[0] + right_eye_width // 2, right_eye_center[1]),
                             (0, 0, 255), 2)
                else:
                    # Rysuj otwarte oko (kontur)
                    cv2.polylines(img_copy, [np.array(right_eye_points)], True, (0, 0, 255), 2)

                    # Rysuj źrenicę
                    right_eye_center = np.mean(np.array(right_eye_points), axis=0).astype(int)
                    pupil_radius = max(1, int((np.max(np.array(right_eye_points)[:, 0]) -
                                               np.min(np.array(right_eye_points)[:, 0])) * 0.15))
                    cv2.circle(img_copy, tuple(right_eye_center), pupil_radius, (0, 0, 0), -1)

            # Rysowanie brwi
            left_eyebrow_points = []
            for idx in self.LEFT_EYEBROW:
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                left_eyebrow_points.append((x, y))

            right_eyebrow_points = []
            for idx in self.RIGHT_EYEBROW:
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                right_eyebrow_points.append((x, y))

            if left_eyebrow_points:
                cv2.polylines(img_copy, [np.array(left_eyebrow_points)], False, (255, 0, 0), 2)

            if right_eyebrow_points:
                cv2.polylines(img_copy, [np.array(right_eyebrow_points)], False, (255, 0, 0), 2)

            # Rysowanie ust
            outer_lips_points = []
            for idx in self.LIPS_OUTER:
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                outer_lips_points.append((x, y))

            inner_lips_points = []
            for idx in self.LIPS_INNER:
                x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                inner_lips_points.append((x, y))

            if outer_lips_points:
                # Rysujemy usta z uwzględnieniem uśmiechu i otwarcia
                mouth_color = (0, 0, 255)  # Czerwony

                # Zewnętrzny kontur ust
                cv2.polylines(img_copy, [np.array(outer_lips_points)], True, mouth_color, 2)

                # Wewnętrzny kontur ust - tylko jeśli usta są otwarte
                if expressions["mouth_open"] > 0.1 and inner_lips_points:
                    cv2.polylines(img_copy, [np.array(inner_lips_points)], True, mouth_color, 1)
                    # Przy bardzo otwartych ustach, wypełniamy wnętrze ciemniejszym kolorem
                    if expressions["mouth_open"] > 0.3:
                        cv2.fillPoly(img_copy, [np.array(inner_lips_points)], (0, 0, 100))  # Ciemny czerwony

            # Rysowanie nosa
            if self.NOSE_BRIDGE and len(self.NOSE_BRIDGE) > 0:
                nose_points = []
                for idx in self.NOSE_BRIDGE:
                    x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
                    nose_points.append((x, y))

                # Dodajemy czubek nosa
                nose_tip_x = int(landmarks[self.NOSE_TIP][0] * w)
                nose_tip_y = int(landmarks[self.NOSE_TIP][1] * h)
                nose_points.append((nose_tip_x, nose_tip_y))

                # Rysujemy linię nosa
                if len(nose_points) > 1:
                    for i in range(len(nose_points) - 1):
                        cv2.line(img_copy, nose_points[i], nose_points[i + 1], (0, 255, 255), 2)

        except Exception as e:
            self.logger.error(
                "FaceMeshDetector",
                f"Błąd podczas rysowania uproszczonej twarzy: {str(e)}",
                log_type="FACE"
            )

        return img_copy

    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Zwraca statystyki detekcji.

        Returns:
            Dict[str, Any]: Statystyki detekcji
        """
        detection_ratio = self.detection_count / max(1, self.frame_count)

        return {
            "total_frames": self.frame_count,
            "detection_count": self.detection_count,
            "detection_ratio": detection_ratio,
            "last_detection_score": self.last_detection_score,
            "max_num_faces": self.max_num_faces,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence
        }

    def reset_stats(self) -> None:
        """
        Resetuje statystyki detekcji.
        """
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_score = 0.0
        self.last_face_data = None

    def close(self) -> None:
        """
        Zwalnia zasoby używane przez detektor.
        """
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
                self.face_mesh = None  # Ustawiamy na None po zamknięciu
                self.logger.debug("FaceMeshDetector", "Detektor twarzy zamknięty", log_type="FACE")
        except Exception as e:
            # Nie zgłaszamy wyjątku, tylko logujemy błąd
            self.logger.warning(
                "FaceMeshDetector",
                f"Błąd podczas zamykania detektora twarzy: {str(e)}",
                log_type="FACE"
            )

    def __del__(self):
        """
        Destruktor klasy, zapewniający zwolnienie zasobów.
        """
        try:
            # Wywołujemy close() tylko jeśli face_mesh nie jest None
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.close()
        except:
            # Ignorujemy błędy w destruktorze - nie ma sensu ich logować,
            # bo logger może już być niedostępny
            pass
