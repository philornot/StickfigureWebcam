#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from src.pose.pose_detector import PoseDetector
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class PostureAnalyzer:
    """
    Klasa do analizy postawy ciała (np. czy osoba siedzi czy stoi).
    Wykorzystuje punkty charakterystyczne wykryte przez PoseDetector
    do określenia pozycji użytkownika.
    """

    def __init__(
            self,
            standing_hip_threshold: float = 0.7,
            confidence_threshold: float = 0.6,
            smoothing_factor: float = 0.7,
            temporal_smoothing: int = 5,
            logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja analizatora postawy.

        Args:
            standing_hip_threshold (float): Próg wysokości bioder dla pozycji stojącej
                                         (jako proporcja wysokości obrazu od góry)
            confidence_threshold (float): Minimalny poziom pewności dla punktów (0.0-1.0)
            smoothing_factor (float): Współczynnik wygładzania detekcji (0.0-1.0)
                                     (wyższe wartości = wolniejsze zmiany)
            temporal_smoothing (int): Liczba klatek używanych do wygładzania czasowego
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("PostureAnalyzer")

        # Parametry detekcji
        self.standing_hip_threshold = standing_hip_threshold
        self.confidence_threshold = confidence_threshold
        self.smoothing_factor = smoothing_factor
        self.temporal_smoothing = temporal_smoothing

        # Stan detekcji
        self.is_sitting = None  # Na początku nie wiemy, potrzebujemy kilku klatek
        self.sitting_probability = 0.5  # Zaczynamy od 50% pewności
        self.history_buffer = []  # Historia detekcji dla wygładzania czasowego
        self.consecutive_frames = 0  # Liczba kolejnych klatek z tą samą detekcją

        # Stałe - indeksy punktów z PoseDetector
        self.LEFT_SHOULDER = PoseDetector.LEFT_SHOULDER
        self.RIGHT_SHOULDER = PoseDetector.RIGHT_SHOULDER
        self.LEFT_HIP = PoseDetector.LEFT_HIP
        self.RIGHT_HIP = PoseDetector.RIGHT_HIP
        self.LEFT_KNEE = PoseDetector.LEFT_KNEE
        self.RIGHT_KNEE = PoseDetector.RIGHT_KNEE
        self.LEFT_ANKLE = PoseDetector.LEFT_ANKLE
        self.RIGHT_ANKLE = PoseDetector.RIGHT_ANKLE

        self.logger.info(
            "PostureAnalyzer",
            f"Analizator postawy zainicjalizowany (próg stania: {standing_hip_threshold})",
            log_type="POSE"
        )

    def analyze_posture(
            self,
            landmarks: List[Tuple[float, float, float, float]],
            frame_height: int,
            frame_width: int
    ) -> Dict[str, Any]:
        """
        Analizuje postawę użytkownika i określa czy siedzi czy stoi.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów charakterystycznych
            frame_height (int): Wysokość klatki
            frame_width (int): Szerokość klatki

        Returns:
            Dict[str, Any]: Słownik z informacjami o postawie
        """
        self.performance.start_timer()

        result = {
            "is_sitting": None,
            "sitting_probability": 0.0,
            "confidence": 0.0,
            "posture": "unknown",
            "visible_keypoints": 0
        }

        if landmarks is None or len(landmarks) < 33:  # MediaPipe Pose ma 33 punkty
            self.logger.debug(
                "PostureAnalyzer",
                "Za mało punktów do analizy postawy",
                log_type="POSE"
            )
            return result

        try:
            # Sprawdzamy widoczność kluczowych punktów
            visible_keypoints = self._count_visible_keypoints(landmarks)
            result["visible_keypoints"] = visible_keypoints

            # Zbyt mało widocznych punktów
            if visible_keypoints < 15:  # Arbitralna wartość - mniej niż połowa punktów
                self.logger.debug(
                    "PostureAnalyzer",
                    f"Za mało widocznych punktów ({visible_keypoints}/33)",
                    log_type="POSE"
                )
                return result

            # Obliczamy prawdopodobieństwo siedzenia na podstawie kilku heurystyk
            hip_score = self._analyze_hip_position(landmarks, frame_height)
            leg_score = self._analyze_leg_visibility(landmarks)
            torso_score = self._analyze_torso_proportion(landmarks)

            # Ważenie poszczególnych wyników
            # Dajemy największą wagę pozycji bioder
            sitting_probability = (0.6 * hip_score + 0.25 * leg_score + 0.15 * torso_score)

            # Ograniczamy do zakresu 0.0-1.0
            sitting_probability = max(0.0, min(1.0, sitting_probability))

            # Stosujemy wygładzanie wykładnicze dla bieżącej detekcji
            if self.sitting_probability is not None:
                sitting_probability = (self.smoothing_factor * self.sitting_probability +
                                       (1 - self.smoothing_factor) * sitting_probability)

            # Dodajemy do historii dla wygładzania czasowego
            self.history_buffer.append(sitting_probability)
            if len(self.history_buffer) > self.temporal_smoothing:
                self.history_buffer.pop(0)

            # Obliczamy wygładzoną wartość
            smoothed_probability = sum(self.history_buffer) / len(self.history_buffer)
            self.sitting_probability = smoothed_probability

            # Określamy stan na podstawie wygładzonej wartości
            is_sitting = smoothed_probability >= 0.5

            # Śledzimy zmiany stanu
            if self.is_sitting is None:
                self.is_sitting = is_sitting
                self.consecutive_frames = 1
            elif self.is_sitting == is_sitting:
                self.consecutive_frames += 1
            else:
                # Zmiana stanu tylko po przekroczeniu progu
                if self.consecutive_frames >= 3:  # Wymagamy min. 3 klatki z tym samym stanem
                    self.is_sitting = is_sitting

                    # Logowanie zmiany stanu
                    self.logger.info(
                        "PostureAnalyzer",
                        f"Zmiana postawy: {'siedząca' if is_sitting else 'stojąca'} "
                        f"(pewność: {smoothed_probability:.2f})",
                        log_type="POSE"
                    )

                self.consecutive_frames = 1

            # Wypełniamy wynik
            result["is_sitting"] = self.is_sitting
            result["sitting_probability"] = smoothed_probability
            result["confidence"] = self._calculate_confidence(landmarks)
            result["posture"] = "sitting" if self.is_sitting else "standing"

            # Dodatkowe informacje dla debugowania
            result["debug"] = {
                "hip_score": hip_score,
                "leg_score": leg_score,
                "torso_score": torso_score,
                "consecutive_frames": self.consecutive_frames
            }

            self.performance.stop_timer()
            return result

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "PostureAnalyzer",
                f"Błąd podczas analizy postawy: {str(e)}",
                log_type="POSE",
                error={"error": str(e)}
            )
            return result

    def _analyze_hip_position(
            self,
            landmarks: List[Tuple[float, float, float, float]],
            frame_height: int
    ) -> float:
        """
        Analizuje pozycję bioder względem wysokości obrazu.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów
            frame_height (int): Wysokość klatki

        Returns:
            float: Wynik 0.0-1.0, gdzie wyższe wartości oznaczają większe prawdopodobieństwo siedzenia
        """
        # Pozycja bioder
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        # Sprawdzamy widoczność
        if left_hip[3] < self.confidence_threshold and right_hip[3] < self.confidence_threshold:
            return 0.5  # Neutralny wynik jeśli biodra nie są widoczne

        # Używamy biodra o lepszej widoczności
        hip_y = left_hip[1] if left_hip[3] > right_hip[3] else right_hip[1]

        # Jeśli biodra są w dolnej części obrazu, jest większa szansa że osoba siedzi
        # Im wyżej biodra, tym większa szansa że osoba stoi

        # hip_y to wartość od 0 (góra) do 1 (dół)
        # Dla postawy stojącej hip_y powinno być mniejsze (biodra wyżej)
        # Dla postawy siedzącej hip_y powinno być większe (biodra niżej)

        # Porównujemy z progiem
        if hip_y < self.standing_hip_threshold:
            # Biodra są wyżej niż próg - prawdopodobnie stoi
            # Im wyżej biodra, tym mniejsza wartość
            return max(0, 1 - (self.standing_hip_threshold - hip_y) / self.standing_hip_threshold)
        else:
            # Biodra są niżej niż próg - prawdopodobnie siedzi
            # Im niżej biodra, tym większa wartość
            return min(1, (hip_y - self.standing_hip_threshold) / (1 - self.standing_hip_threshold) * 0.8 + 0.2)

    def _analyze_leg_visibility(self, landmarks: List[Tuple[float, float, float, float]]) -> float:
        """
        Analizuje widoczność nóg - przy siedzeniu często nie są widoczne.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            float: Wynik 0.0-1.0, gdzie wyższe wartości oznaczają większe prawdopodobieństwo siedzenia
        """
        # Sprawdzamy widoczność kostek i kolan
        left_ankle_visible = landmarks[self.LEFT_ANKLE][3] > self.confidence_threshold
        right_ankle_visible = landmarks[self.RIGHT_ANKLE][3] > self.confidence_threshold
        left_knee_visible = landmarks[self.LEFT_KNEE][3] > self.confidence_threshold
        right_knee_visible = landmarks[self.RIGHT_KNEE][3] > self.confidence_threshold

        # Liczymy widoczne części nóg
        visible_parts = sum([left_ankle_visible, right_ankle_visible, left_knee_visible, right_knee_visible])

        # Im mniej widocznych części nóg, tym większe prawdopodobieństwo siedzenia
        if visible_parts == 0:
            return 0.9  # Wysokie prawdopodobieństwo siedzenia gdy nie widać nóg
        elif visible_parts == 1:
            return 0.7
        elif visible_parts == 2:
            return 0.5  # Neutralne
        elif visible_parts == 3:
            return 0.3
        else:  # visible_parts == 4
            return 0.1  # Niskie prawdopodobieństwo siedzenia gdy widać wszystkie części nóg

    def _analyze_torso_proportion(self, landmarks: List[Tuple[float, float, float, float]]) -> float:
        """
        Analizuje proporcje tułowia (odległość biodra-ramię vs biodro-kolano).

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            float: Wynik 0.0-1.0, gdzie wyższe wartości oznaczają większe prawdopodobieństwo siedzenia
        """
        # Sprawdzamy widoczność punktów
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        left_knee = landmarks[self.LEFT_KNEE]
        right_knee = landmarks[self.RIGHT_KNEE]

        # Sprawdzamy czy mamy wystarczająco punktów do analizy
        left_side_visible = (left_shoulder[3] > self.confidence_threshold and
                             left_hip[3] > self.confidence_threshold and
                             left_knee[3] > self.confidence_threshold)

        right_side_visible = (right_shoulder[3] > self.confidence_threshold and
                              right_hip[3] > self.confidence_threshold and
                              right_knee[3] > self.confidence_threshold)

        if not left_side_visible and not right_side_visible:
            return 0.5  # Neutralny wynik jeśli nie mamy wystarczających danych

        # Wybieramy stronę z lepszą widocznością
        if left_side_visible and (not right_side_visible or
                                  (left_shoulder[3] + left_hip[3] + left_knee[3]) >
                                  (right_shoulder[3] + right_hip[3] + right_knee[3])):
            shoulder = left_shoulder
            hip = left_hip
            knee = left_knee
        else:
            shoulder = right_shoulder
            hip = right_hip
            knee = right_knee

        # Obliczamy długości
        torso_length = np.sqrt((shoulder[0] - hip[0]) ** 2 + (shoulder[1] - hip[1]) ** 2)
        upper_leg_length = np.sqrt((hip[0] - knee[0]) ** 2 + (hip[1] - knee[1]) ** 2)

        # Jeśli któraś z długości jest bliska zeru, zwracamy neutralny wynik
        if torso_length < 0.01 or upper_leg_length < 0.01:
            return 0.5

        # Obliczamy stosunek długości tułowia do długości górnej części nogi
        ratio = torso_length / upper_leg_length

        # Interpretacja proporcji:
        # - W postawie stojącej stosunek tułowia do nogi jest zazwyczaj mniejszy (ok. 0.8-1.2)
        # - W postawie siedzącej stosunek może być większy (ok. 1.5-2.5)

        if ratio < 0.8:
            return 0.2  # Bardzo prawdopodobne, że stoi
        elif ratio < 1.2:
            return 0.3  # Prawdopodobnie stoi
        elif ratio < 1.5:
            return 0.6  # Lekkie wskazanie na siedzenie
        elif ratio < 2.5:
            return 0.8  # Prawdopodobnie siedzi
        else:
            return 0.7  # Bardzo duży stosunek może wskazywać na błąd detekcji

    def _count_visible_keypoints(self, landmarks: List[Tuple[float, float, float, float]]) -> int:
        """
        Liczy liczbę widocznych punktów charakterystycznych.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            int: Liczba widocznych punktów
        """
        visible_count = 0
        for _, _, _, visibility in landmarks:
            if visibility > self.confidence_threshold:
                visible_count += 1

        return visible_count

    def _calculate_confidence(self, landmarks: List[Tuple[float, float, float, float]]) -> float:
        """
        Oblicza pewność detekcji postawy na podstawie widoczności kluczowych punktów.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            float: Pewność detekcji (0.0-1.0)
        """
        # Kluczowe punkty dla detekcji postawy
        key_points = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_KNEE, self.RIGHT_KNEE,
            self.LEFT_ANKLE, self.RIGHT_ANKLE
        ]

        # Średnia widoczność kluczowych punktów
        total_visibility = 0.0
        for point_id in key_points:
            if point_id < len(landmarks):
                total_visibility += landmarks[point_id][3]

        avg_visibility = total_visibility / len(key_points)

        # Uwzględniamy też liczbę klatek konsekwentnej detekcji
        temporal_factor = min(1.0, self.consecutive_frames / 10.0)

        # Łączymy oba czynniki
        confidence = avg_visibility * 0.7 + temporal_factor * 0.3

        return confidence

    def reset(self) -> None:
        """
        Resetuje stan analizatora.
        """
        self.is_sitting = None
        self.sitting_probability = 0.5
        self.history_buffer = []
        self.consecutive_frames = 0

        self.logger.debug("PostureAnalyzer", "Stan analizatora postawy zresetowany", log_type="POSE")

    def get_current_posture(self) -> Dict[str, Any]:
        """
        Zwraca bieżący stan postawy.

        Returns:
            Dict[str, Any]: Słownik z informacjami o bieżącej postawie
        """
        return {
            "is_sitting": self.is_sitting,
            "posture": "sitting" if self.is_sitting else ("standing" if self.is_sitting is not None else "unknown"),
            "sitting_probability": self.sitting_probability,
            "consecutive_frames": self.consecutive_frames
        }

    def update_thresholds(
            self,
            standing_hip_threshold: Optional[float] = None,
            confidence_threshold: Optional[float] = None,
            smoothing_factor: Optional[float] = None
    ) -> None:
        """
        Aktualizuje progi detekcji.

        Args:
            standing_hip_threshold (Optional[float]): Nowy próg wysokości bioder
            confidence_threshold (Optional[float]): Nowy próg pewności
            smoothing_factor (Optional[float]): Nowy współczynnik wygładzania
        """
        if standing_hip_threshold is not None:
            self.standing_hip_threshold = standing_hip_threshold

        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

        if smoothing_factor is not None:
            self.smoothing_factor = smoothing_factor

        self.logger.info(
            "PostureAnalyzer",
            f"Zaktualizowano parametry analizatora (hip_threshold={self.standing_hip_threshold}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"smoothing_factor={self.smoothing_factor})",
            log_type="POSE"
        )
