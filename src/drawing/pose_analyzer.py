#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/pose_analyzer.py

from typing import List, Tuple, Optional


class PoseAnalyzer:
    """
    Klasa do analizy pozy człowieka na podstawie punktów charakterystycznych.
    Koncentruje się na rozróżnianiu pozycji siedzącej i stojącej.
    """

    # Indeksy punktów z MediaPipe Pose
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(self, sitting_threshold: float = 0.3):
        """
        Inicjalizacja analizatora pozy.

        Args:
            sitting_threshold (float): Próg określający stosunek pozycji bioder
                                     do wysokości ramion dla rozpoznania siedzenia
        """
        self.sitting_threshold = sitting_threshold

    def is_sitting(
        self,
        landmarks: Optional[List[Tuple[float, float, float, float]]],
        frame_height: Optional[int] = None,
        frame_width: Optional[int] = None
    ) -> bool:
        """
        Określa, czy osoba siedzi na podstawie punktów charakterystycznych.

        Args:
            landmarks (Optional[List[Tuple[float, float, float, float]]]): Lista punktów (x, y, z, visibility)
            frame_height (Optional[int]): Wysokość klatki dla dodatkowej analizy
            frame_width (Optional[int]): Szerokość klatki dla dodatkowej analizy

        Returns:
            bool: True jeśli osoba siedzi, False w przeciwnym przypadku
        """
        # Jeśli nie ma punktów charakterystycznych, zakładamy, że osoba stoi
        if landmarks is None or len(landmarks) < 29:  # Potrzebujemy co najmniej punktów kostek (indeks 28)
            return False

        try:
            # Pobieramy pozycje kluczowych punktów
            hips_visible = False
            knees_visible = False

            # Sprawdzamy widoczność bioder
            left_hip_visible = len(landmarks) > self.LEFT_HIP and landmarks[self.LEFT_HIP][3] > 0.5
            right_hip_visible = len(landmarks) > self.RIGHT_HIP and landmarks[self.RIGHT_HIP][3] > 0.5

            if left_hip_visible or right_hip_visible:
                hips_visible = True

            # Sprawdzamy widoczność kolan
            left_knee_visible = len(landmarks) > self.LEFT_KNEE and landmarks[self.LEFT_KNEE][3] > 0.5
            right_knee_visible = len(landmarks) > self.RIGHT_KNEE and landmarks[self.RIGHT_KNEE][3] > 0.5

            if left_knee_visible or right_knee_visible:
                knees_visible = True

            # Pobieramy pozycje punktów
            # Sprawdzamy widoczność barków
            shoulders_y = None
            if len(landmarks) > self.LEFT_SHOULDER and landmarks[self.LEFT_SHOULDER][3] > 0.5:
                shoulders_y = landmarks[self.LEFT_SHOULDER][1]
            elif len(landmarks) > self.RIGHT_SHOULDER and landmarks[self.RIGHT_SHOULDER][3] > 0.5:
                shoulders_y = landmarks[self.RIGHT_SHOULDER][1]

            # Sprawdzamy pozycje bioder
            hips_y = None
            if left_hip_visible:
                hips_y = landmarks[self.LEFT_HIP][1]
            elif right_hip_visible:
                hips_y = landmarks[self.RIGHT_HIP][1]

            # Jeśli nie mamy tych podstawowych punktów, nie możemy określić pozy
            if shoulders_y is None or hips_y is None:
                return False  # Domyślnie zakładamy pozycję stojącą

            # Główne kryterium: porównanie pozycji bioder do barków
            # Gdy biodra są znacznie niżej niż barki, to prawdopodobnie osoba siedzi
            hip_shoulder_ratio = hips_y - shoulders_y

            # Ustalamy próg siedzenia/stania
            sitting_threshold = self.sitting_threshold

            # Heurystyka:
            # 1. Jeśli biodra są znacznie niżej niż barki, to prawdopodobnie osoba siedzi
            is_sitting_by_position = hip_shoulder_ratio > sitting_threshold

            # 2. Jeśli kolana są widoczne i są blisko bioder (w pionie), to też prawdopodobnie siedzi
            is_sitting_by_knees = False
            if knees_visible and hips_visible:
                knees_y = None
                if left_knee_visible:
                    knees_y = landmarks[self.LEFT_KNEE][1]
                elif right_knee_visible:
                    knees_y = landmarks[self.RIGHT_KNEE][1]

                if knees_y is not None:
                    # Odległość między biodrami a kolanami
                    hip_knee_distance = abs(knees_y - hips_y)
                    if hip_knee_distance < 0.15:  # Bliska odległość wskazuje na siedzenie
                        is_sitting_by_knees = True

            # 3. Sprawdzamy, czy kostki są widoczne - jeśli nie, to prawdopodobnie siedzi
            ankles_visible = (len(landmarks) > self.LEFT_ANKLE and landmarks[self.LEFT_ANKLE][3] > 0.5) or \
                             (len(landmarks) > self.RIGHT_ANKLE and landmarks[self.RIGHT_ANKLE][3] > 0.5)

            # Końcowa decyzja - siedzi jeśli spełnione jest którekolwiek z kryteriów
            return is_sitting_by_position or is_sitting_by_knees or not ankles_visible

        except Exception as e:
            print(f"Błąd podczas analizy pozy: {str(e)}")
            return False  # W razie błędu zakładamy, że osoba stoi
