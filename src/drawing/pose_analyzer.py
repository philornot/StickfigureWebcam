#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# drawing/pose_analyzer.py

from typing import List, Tuple, Optional


class PoseAnalyzer:
    """
    Klasa do analizy pozy człowieka na podstawie punktów charakterystycznych.
    Koncentruje się na rozróżnianiu pozycji siedzącej i stojącej.
    """

    # Indeksy punktów z MediaPipe Pose
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14

    def __init__(self, sitting_threshold: float = 0.3):
        """
        Inicjalizacja analizatora pozy.

        Args:
            sitting_threshold (float): Próg określający stosunek pozycji bioder
                                     do wysokości ramion dla rozpoznania siedzenia
        """
        self.sitting_threshold = sitting_threshold
        print(f"PoseAnalyzer zainicjalizowany (próg siedzenia: {sitting_threshold})")

    def is_sitting(
        self,
        landmarks: List[Tuple[float, float, float, float]],
        frame_height: Optional[int] = None
    ) -> bool:
        """
        Określa, czy osoba siedzi na podstawie punktów charakterystycznych.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów (x, y, z, visibility)
            frame_height (Optional[int]): Wysokość klatki dla dodatkowej analizy

        Returns:
            bool: True jeśli osoba siedzi, False w przeciwnym przypadku
        """
        if landmarks is None or len(landmarks) < 15:  # Potrzebujemy co najmniej punktów kolan
            return False

        try:
            # Pobieramy punkty do analizy
            shoulders_y = self._get_shoulders_y(landmarks)
            hips_y = self._get_hips_y(landmarks)
            knees_y = self._get_knees_y(landmarks)

            if shoulders_y is None or hips_y is None:
                return False

            # Podstawowa heurystyka siedzenia
            # 1. Sprawdzamy czy biodra są blisko kolan (w pionie)
            hip_knee_ratio = self._calculate_hip_knee_ratio(hips_y, knees_y)

            # 2. Sprawdzamy czy biodra są niżej niż oczekiwana pozycja dla stania
            # (biodra podczas siedzenia są zazwyczaj niżej w stosunku do ramion)
            shoulder_hip_ratio = (hips_y - shoulders_y)

            # 3. Sprawdzamy położenie bioder względem dolnej krawędzi kadru (jeśli podano wysokość)
            bottom_proximity = self._calculate_bottom_proximity(hips_y, frame_height)

            # Kombinacja powyższych czynników dla precyzyjniejszej detekcji
            if hip_knee_ratio < self.sitting_threshold:
                # Biodra blisko kolan - prawdopodobnie siedzi
                sitting_confidence = 0.8
            else:
                # Biodra daleko od kolan - prawdopodobnie stoi
                sitting_confidence = 0.2

            # Jeśli biodra są znacznie niżej niż ramiona, zwiększamy pewność siedzenia
            if shoulder_hip_ratio > 0.3:
                sitting_confidence += 0.15

            # Jeśli biodra są blisko dolnej krawędzi kadru, zwiększamy pewność siedzenia
            if bottom_proximity and bottom_proximity > 0.8:
                sitting_confidence += 0.1

            return sitting_confidence > 0.6  # Próg pewności

        except Exception as e:
            print(f"Błąd podczas analizy pozy: {str(e)}")
            return False  # W razie błędu zakładamy, że osoba stoi

    def _get_shoulders_y(self, landmarks: List[Tuple[float, float, float, float]]) -> Optional[float]:
        """
        Zwraca średnią pozycję Y ramion.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            Optional[float]: Średnia pozycja Y ramion lub None jeśli nie wykryto
        """
        left_shoulder = landmarks[self.LEFT_SHOULDER] if self.LEFT_SHOULDER < len(landmarks) else None
        right_shoulder = landmarks[self.RIGHT_SHOULDER] if self.RIGHT_SHOULDER < len(landmarks) else None

        if left_shoulder and left_shoulder[3] > 0.5 and right_shoulder and right_shoulder[3] > 0.5:
            # Mamy oba ramiona z dobrą widocznością
            return (left_shoulder[1] + right_shoulder[1]) / 2
        elif left_shoulder and left_shoulder[3] > 0.5:
            # Mamy tylko lewe ramię
            return left_shoulder[1]
        elif right_shoulder and right_shoulder[3] > 0.5:
            # Mamy tylko prawe ramię
            return right_shoulder[1]
        else:
            return None

    def _get_hips_y(self, landmarks: List[Tuple[float, float, float, float]]) -> Optional[float]:
        """
        Zwraca średnią pozycję Y bioder.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            Optional[float]: Średnia pozycja Y bioder lub None jeśli nie wykryto
        """
        left_hip = landmarks[self.LEFT_HIP] if self.LEFT_HIP < len(landmarks) else None
        right_hip = landmarks[self.RIGHT_HIP] if self.RIGHT_HIP < len(landmarks) else None

        if left_hip and left_hip[3] > 0.5 and right_hip and right_hip[3] > 0.5:
            # Mamy oba biodra z dobrą widocznością
            return (left_hip[1] + right_hip[1]) / 2
        elif left_hip and left_hip[3] > 0.5:
            # Mamy tylko lewe biodro
            return left_hip[1]
        elif right_hip and right_hip[3] > 0.5:
            # Mamy tylko prawe biodro
            return right_hip[1]
        else:
            return None

    def _get_knees_y(self, landmarks: List[Tuple[float, float, float, float]]) -> Optional[float]:
        """
        Zwraca średnią pozycję Y kolan.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów

        Returns:
            Optional[float]: Średnia pozycja Y kolan lub None jeśli nie wykryto
        """
        left_knee = landmarks[self.LEFT_KNEE] if self.LEFT_KNEE < len(landmarks) else None
        right_knee = landmarks[self.RIGHT_KNEE] if self.RIGHT_KNEE < len(landmarks) else None

        if left_knee and left_knee[3] > 0.5 and right_knee and right_knee[3] > 0.5:
            # Mamy oba kolana z dobrą widocznością
            return (left_knee[1] + right_knee[1]) / 2
        elif left_knee and left_knee[3] > 0.5:
            # Mamy tylko lewe kolano
            return left_knee[1]
        elif right_knee and right_knee[3] > 0.5:
            # Mamy tylko prawe kolano
            return right_knee[1]
        else:
            return None

    def _calculate_hip_knee_ratio(
        self,
        hips_y: Optional[float],
        knees_y: Optional[float]
    ) -> float:
        """
        Oblicza stosunek odległości między biodrami a kolanami.

        Args:
            hips_y (Optional[float]): Pozycja Y bioder
            knees_y (Optional[float]): Pozycja Y kolan

        Returns:
            float: Stosunek odległości lub wartość domyślna (0.5) jeśli brak danych
        """
        if hips_y is None or knees_y is None:
            return 0.5  # Wartość domyślna

        # Odległość między biodrami a kolanami
        distance = abs(knees_y - hips_y)

        # Normalizacja do zakresu 0-1, gdzie mniejsze wartości oznaczają bliskość (siedzenie)
        # a większe - odległość (stanie)
        if distance < 0.05:  # Bardzo bliskie punkty
            return 0.1  # Prawie na pewno siedzi
        elif distance > 0.3:  # Dalekie punkty
            return 0.9  # Prawie na pewno stoi
        else:
            # Liniowa interpolacja między skrajnymi przypadkami
            return 0.1 + ((distance - 0.05) / 0.25) * 0.8

    def _calculate_bottom_proximity(
        self,
        hips_y: Optional[float],
        frame_height: Optional[int]
    ) -> Optional[float]:
        """
        Oblicza bliskość bioder do dolnej krawędzi kadru.

        Args:
            hips_y (Optional[float]): Pozycja Y bioder (wartość znormalizowana 0-1)
            frame_height (Optional[int]): Wysokość klatki w pikselach

        Returns:
            Optional[float]: Wartość bliskości (0-1) lub None jeśli brak danych
        """
        if hips_y is None or frame_height is None:
            return None

        # Wartość hips_y jest już znormalizowana do zakresu 0-1
        # Im bliżej 1, tym bliżej dolnej krawędzi
        # Dla siedzenia biodra są często bliżej dolnej krawędzi kadru
        bottom_proximity = hips_y

        return bottom_proximity
