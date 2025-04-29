#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Optional, Any

import cv2
import mediapipe as mp
import numpy as np

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class PoseDetector:
    """
    Klasa do wykrywania pozy ciała przy użyciu MediaPipe Pose.
    Wykrywa 33 punkty charakterystyczne ciała (landmarks) i dostarcza ich
    współrzędne oraz wizualizację.
    """

    # Indeksy ważnych punktów charakterystycznych (landmarks)
    NOSE = 0

    # Punkty twarzy
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10

    # Punkty tułowia
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    # Punkty bioder i nóg
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # Definiujemy połączenia między punktami dla wizualizacji
    POSE_CONNECTIONS = [
        # Głowa
        (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
        (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),

        # Tułów
        (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),

        # Nogi
        (LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]

    def __init__(
            self,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
            model_complexity: int = 1,
            smooth_landmarks: bool = True,
            enable_segmentation: bool = False,
            logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja detektora pozy.

        Args:
            min_detection_confidence (float): Minimalna pewność detekcji pozy (0.0-1.0)
            min_tracking_confidence (float): Minimalna pewność śledzenia pozy (0.0-1.0)
            model_complexity (int): Złożoność modelu (0, 1, 2) - wyższe wartości to
                                   większa dokładność, ale wolniejsze działanie
            smooth_landmarks (bool): Czy wygładzać ruchy punktów charakterystycznych
            enable_segmentation (bool): Czy włączyć segmentację człowieka od tła
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("PoseDetector")

        # Parametry detekcji
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation

        # Statystyki detekcji
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_score = 0.0

        # Inicjalizacja MediaPipe Pose
        self.logger.debug(
            "PoseDetector",
            f"Inicjalizacja MediaPipe Pose (model_complexity={model_complexity})",
            log_type="POSE"
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Inicjalizacja detektora pozy
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.logger.info(
            "PoseDetector",
            "MediaPipe Pose zainicjalizowany pomyślnie",
            log_type="POSE"
        )

    def detect_pose(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Wykrywa pozę na obrazie.

        Args:
            image (np.ndarray): Obraz wejściowy w formacie BGR (OpenCV)

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - bool: True jeśli wykryto pozę, False w przeciwnym razie
                - Dict: Słownik zawierający informacje o wykrytej pozie
        """
        self.performance.start_timer()
        self.frame_count += 1

        # MediaPipe wymaga obrazu w formacie RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detekcja pozy
        try:
            results = self.pose.process(image_rgb)

            # Przygotowanie wyników
            pose_data = {
                "landmarks": None,
                "world_landmarks": None,
                "segmentation_mask": None,
                "detection_score": 0.0,
                "has_pose": False,
                "frame_height": image.shape[0],
                "frame_width": image.shape[1]
            }

            # Sprawdzenie czy poza została wykryta
            if results.pose_landmarks:
                # Poza wykryta
                self.detection_count += 1
                pose_data["has_pose"] = True

                # Konwersja punktów charakterystycznych na listę krotek (x, y, z, visibility)
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))

                pose_data["landmarks"] = landmarks

                # Przechowujemy również punkty 3D (world landmarks)
                if results.pose_world_landmarks:
                    world_landmarks = []
                    for landmark in results.pose_world_landmarks.landmark:
                        world_landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))

                    pose_data["world_landmarks"] = world_landmarks

                # Wyciągamy wynik segmentacji jeśli jest dostępny
                if self.enable_segmentation and results.segmentation_mask is not None:
                    pose_data["segmentation_mask"] = results.segmentation_mask

                # Szacowanie pewności detekcji (średnia widoczność punktów)
                visibilities = [landmark[3] for landmark in landmarks]
                pose_data["detection_score"] = sum(visibilities) / len(visibilities)
                self.last_detection_score = pose_data["detection_score"]

                # Logowanie co 30 klatek
                if self.frame_count % 30 == 0:
                    self.logger.debug(
                        "PoseDetector",
                        f"Wykryto pozę z pewnością {self.last_detection_score:.2f}",
                        log_type="POSE"
                    )

                # Monitorowanie wydajności
                self.performance.stop_timer()
                execution_time = self.performance.get_last_execution_time() * 1000  # ms

                # Statystyki wydajności co 100 detekcji
                if self.detection_count % 100 == 0:
                    self.logger.info(
                        "PoseDetector",
                        f"Wykryto {self.detection_count} póz, "
                        f"stosunek detekcji: {self.detection_count / self.frame_count:.2f}",
                        log_type="POSE"
                    )
                    self.logger.performance_metrics(0, execution_time, "PoseDetector")

                return True, pose_data
            else:
                # Brak detekcji pozy
                self.performance.stop_timer()

                # Logowanie braku detekcji co 50 klatek bez detekcji
                if self.frame_count % 50 == 0 and self.detection_count == 0:
                    self.logger.warning(
                        "PoseDetector",
                        "Nie wykryto pozy w ostatnich 50 klatkach",
                        log_type="POSE"
                    )
                elif self.frame_count % 100 == 0:
                    self.logger.debug(
                        "PoseDetector",
                        f"Brak detekcji pozy, stosunek detekcji: {self.detection_count / self.frame_count:.2f}",
                        log_type="POSE"
                    )

                return False, pose_data

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "PoseDetector",
                f"Błąd podczas detekcji pozy: {str(e)}",
                log_type="POSE",
                error={"error": str(e)}
            )
            return False, {"error": str(e), "has_pose": False}

    def draw_pose_on_image(
            self,
            image: np.ndarray,
            landmarks: List[Tuple[float, float, float, float]],
            draw_connections: bool = True,
            keypoint_radius: int = 5,
            keypoint_color: Tuple[int, int, int] = (0, 255, 0),  # BGR: zielony
            connection_color: Tuple[int, int, int] = (255, 255, 0),  # BGR: turkusowy
            connection_thickness: int = 2
    ) -> np.ndarray:
        """
        Rysuje wykrytą pozę na obrazie.

        Args:
            image (np.ndarray): Obraz wejściowy (BGR)
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów (x, y, z, visibility)
            draw_connections (bool): Czy rysować połączenia między punktami
            keypoint_radius (int): Promień punktów
            keypoint_color (Tuple[int, int, int]): Kolor punktów (BGR)
            connection_color (Tuple[int, int, int]): Kolor połączeń (BGR)
            connection_thickness (int): Grubość linii połączeń

        Returns:
            np.ndarray: Obraz z narysowaną pozą
        """
        if landmarks is None or len(landmarks) == 0:
            return image

        # Tworzymy kopię obrazu, aby nie modyfikować oryginału
        img_copy = image.copy()

        h, w, _ = img_copy.shape

        # Rysowanie punktów (keypoints)
        for i, landmark in enumerate(landmarks):
            x, y, _, visibility = landmark

            # Tylko punkty z dobrą widocznością
            if visibility > 0.5:
                # Konwersja współrzędnych względnych na bezwzględne
                cx, cy = int(x * w), int(y * h)

                # Rysowanie punktu
                cv2.circle(img_copy, (cx, cy), keypoint_radius, keypoint_color, -1)

        # Rysowanie połączeń
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection

                # Sprawdzenie czy punkty istnieją i są widoczne
                if (len(landmarks) > start_idx and len(landmarks) > end_idx and
                        landmarks[start_idx][3] > 0.5 and landmarks[end_idx][3] > 0.5):
                    start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                    end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))

                    cv2.line(img_copy, start_point, end_point, connection_color, connection_thickness)

        return img_copy

    def draw_pose_with_mediapipe(self, image: np.ndarray, landmarks: Any) -> np.ndarray:
        """
        Rysuje pozę na obrazie używając wbudowanych funkcji MediaPipe.

        Args:
            image (np.ndarray): Obraz wejściowy (BGR)
            landmarks: Obiekt pose_landmarks z MediaPipe

        Returns:
            np.ndarray: Obraz z narysowaną pozą
        """
        if landmarks is None:
            return image

        # Konwersja listy krotek do formatu MediaPipe jeśli potrzeba
        if isinstance(landmarks, list):
            mp_landmarks = self._convert_to_mp_landmarks(landmarks, image.shape[1], image.shape[0])
        else:
            mp_landmarks = landmarks

        # Tworzymy kopię obrazu
        img_copy = image.copy()

        # Rysowanie pozy przy użyciu MediaPipe
        self.mp_drawing.draw_landmarks(
            img_copy,
            mp_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        return img_copy

    def _convert_to_mp_landmarks(
            self,
            landmarks: List[Tuple[float, float, float, float]],
            img_width: int,
            img_height: int
    ) -> Any:
        """
        Konwertuje listę punktów do formatu MediaPipe.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów (x, y, z, visibility)
            img_width (int): Szerokość obrazu
            img_height (int): Wysokość obrazu

        Returns:
            mp_pose.PoseLandmarkList: Punkty w formacie MediaPipe
        """
        landmark_list = self.mp_pose.PoseLandmarkList()

        for x, y, z, visibility in landmarks:
            landmark = self.mp_pose.PoseLandmark()
            landmark.x = x
            landmark.y = y
            landmark.z = z
            landmark.visibility = visibility
            landmark_list.landmark.append(landmark)

        return landmark_list

    def get_landmark_position(
            self,
            landmarks: List[Tuple[float, float, float, float]],
            landmark_id: int,
            img_width: int,
            img_height: int
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Zwraca pozycję konkretnego punktu.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            landmark_id (int): Indeks punktu (np. PoseDetector.NOSE)
            img_width (int): Szerokość obrazu
            img_height (int): Wysokość obrazu

        Returns:
            Optional[Tuple[int, int, float, float]]:
                (x, y, z, visibility) lub None jeśli punkt nie istnieje
        """
        if landmarks is None or landmark_id >= len(landmarks):
            return None

        x, y, z, visibility = landmarks[landmark_id]
        return (int(x * img_width), int(y * img_height), z, visibility)

    def calculate_angle(
            self,
            landmarks: List[Tuple[float, float, float, float]],
            point1: int,
            point2: int,
            point3: int
    ) -> float:
        """
        Oblicza kąt między trzema punktami.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            point1 (int): Indeks pierwszego punktu
            point2 (int): Indeks drugiego punktu (wierzchołek kąta)
            point3 (int): Indeks trzeciego punktu

        Returns:
            float: Kąt w stopniach (0-180)
        """
        if landmarks is None or point1 >= len(landmarks) or point2 >= len(landmarks) or point3 >= len(landmarks):
            return 0.0

        # Pobieramy współrzędne punktów
        x1, y1, _, _ = landmarks[point1]
        x2, y2, _, _ = landmarks[point2]
        x3, y3, _, _ = landmarks[point3]

        # Obliczamy kąt
        angle_radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle_degrees = np.abs(angle_radians * 180.0 / np.pi)

        # Upewniamy się, że kąt jest w zakresie 0-180
        if angle_degrees > 180.0:
            angle_degrees = 360.0 - angle_degrees

        return angle_degrees

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
            "model_complexity": self.model_complexity,
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

    def close(self) -> None:
        """
        Zwalnia zasoby używane przez detektor.
        """
        if self.pose:
            self.pose.close()
            self.logger.debug("PoseDetector", "Detektor pozy zamknięty", log_type="POSE")

    def __del__(self):
        """
        Destruktor klasy, zapewniający zwolnienie zasobów.
        """
        self.close()
