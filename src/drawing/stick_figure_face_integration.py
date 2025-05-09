#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/stick_figure_face_integration.py

from typing import Tuple, Dict, Any, Optional

import numpy as np

from src.drawing.face_renderer import FaceRenderer
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class StickFigureFaceIntegration:
    """
    Klasa integrująca renderer twarzy FaceMesh z podstawowym rendererem StickFigure.

    Zapewnia płynne przejście między prostym renderowaniem twarzy a bardziej
    szczegółowym wyglądem opartym na danych z MediaPipe FaceMesh.
    """

    def __init__(
        self,
        feature_color: Tuple[int, int, int] = (0, 0, 0),  # Czarny
        detail_level: str = "medium",  # "low", "medium", "high"
        smooth_factor: float = 0.3,
        use_face_mesh: bool = True,
        logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja integratora twarzy.

        Args:
            feature_color (Tuple[int, int, int]): Kolor elementów twarzy (BGR)
            detail_level (str): Poziom szczegółowości twarzy ("low", "medium", "high")
            smooth_factor (float): Współczynnik wygładzania ruchu (0.0-1.0)
            use_face_mesh (bool): Czy używać danych z FaceMesh (jeśli False, używa prostej twarzy)
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("StickFigureFaceIntegration")

        # Parametry
        self.feature_color = feature_color
        self.detail_level = detail_level
        self.smooth_factor = smooth_factor
        self.use_face_mesh = use_face_mesh

        # Inicjalizacja renderera twarzy
        self.face_renderer = FaceRenderer(
            head_radius=30,  # Zostanie zaktualizowane podczas używania
            feature_color=feature_color,
            smooth_factor=smooth_factor,
            detail_level=detail_level,
            logger=logger
        )

        # Flagi stanu
        self.is_initialized = True

        self.logger.info(
            "StickFigureFaceIntegration",
            f"Integrator twarzy zainicjalizowany (detail_level={detail_level}, use_face_mesh={use_face_mesh})",
            log_type="DRAWING"
        )

    def integrate_face(
        self,
        canvas: np.ndarray,
        head_position: Tuple[int, int],
        head_radius: int,
        face_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Integruje twarz z FaceMeshDetector z istniejącym rendererem stick figure.

        Args:
            canvas (np.ndarray): Płótno z już narysowaną podstawową postacią stick figure
            head_position (Tuple[int, int]): Pozycja środka głowy (x, y) w pikselach
            head_radius (int): Promień głowy w pikselach
            face_data (Optional[Dict[str, Any]]): Dane twarzy z FaceMeshDetector
                                               lub None, jeśli dane nie są dostępne

        Returns:
            np.ndarray: Płótno z zintegrowaną twarzą
        """
        self.performance.start_timer()

        try:
            if not self.use_face_mesh or face_data is None or "has_face" not in face_data or not face_data["has_face"]:
                # Jeżeli nie używamy FaceMesh lub dane twarzy są niedostępne,
                # pozostawiamy canvas bez zmian (używamy prostej twarzy z StickFigureRenderer)
                self.performance.stop_timer()
                return canvas

            # Aktualizuj promień głowy w rendererze
            self.face_renderer.set_head_radius(head_radius)

            # Renderuj twarz z uwzględnieniem danych FaceMesh
            result_canvas = self.face_renderer.render_face(canvas, head_position, face_data)

            self.performance.stop_timer()
            return result_canvas

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "StickFigureFaceIntegration",
                f"Błąd podczas integracji twarzy: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)}
            )
            # W przypadku błędu zwróć oryginalny canvas
            return canvas

    def set_use_face_mesh(self, use_face_mesh: bool) -> None:
        """
        Włącza lub wyłącza używanie danych z FaceMesh.

        Args:
            use_face_mesh (bool): Czy używać danych z FaceMesh
        """
        self.use_face_mesh = use_face_mesh
        self.logger.info(
            "StickFigureFaceIntegration",
            f"Zmieniono użycie FaceMesh na: {use_face_mesh}",
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
            self.face_renderer.set_detail_level(detail_level)
            self.logger.info(
                "StickFigureFaceIntegration",
                f"Zmieniono poziom szczegółowości na {detail_level}",
                log_type="DRAWING"
            )
        else:
            self.logger.warning(
                "StickFigureFaceIntegration",
                f"Nieprawidłowy poziom szczegółowości: {detail_level}. Dozwolone wartości: low, medium, high",
                log_type="DRAWING"
            )

    def override_face(
        self,
        canvas: np.ndarray,
        head_position: Tuple[int, int],
        head_radius: int,
        mood: str = "happy"
    ) -> np.ndarray:
        """
        Nadpisuje istniejącą twarz na płótnie ekspresyjną buźką.
        Przydatne do wymuszenia określonego nastroju niezależnie od danych FaceMesh.

        Args:
            canvas (np.ndarray): Płótno z już narysowaną postacią
            head_position (Tuple[int, int]): Pozycja środka głowy (x, y) w pikselach
            head_radius (int): Promień głowy w pikselach
            mood (str): Nastrój do wyrażenia ("happy", "sad", "surprised", "neutral", "wink")

        Returns:
            np.ndarray: Płótno z nadpisaną twarzą
        """
        self.performance.start_timer()

        # Tworzymy sztuczne dane twarzy dla określonego nastroju
        fake_expressions = {
            "mouth_open": 0.0,
            "smile": 0.0,
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0
        }

        # Ustawiamy wartości ekspresji w zależności od nastroju
        if mood == "happy":
            fake_expressions["smile"] = 0.8
        elif mood == "sad":
            fake_expressions["smile"] = 0.2
            fake_expressions["eyebrow_raised"] = 0.2
        elif mood == "surprised":
            fake_expressions["mouth_open"] = 0.7
            fake_expressions["eyebrow_raised"] = 0.8
            fake_expressions["surprise"] = 0.8
        elif mood == "neutral":
            fake_expressions["smile"] = 0.5
        elif mood == "wink":
            fake_expressions["smile"] = 0.7
            fake_expressions["left_eye_open"] = 0.1
        else:
            self.logger.warning(
                "StickFigureFaceIntegration",
                f"Nieznany nastrój: {mood}, używam 'happy'",
                log_type="DRAWING"
            )
            fake_expressions["smile"] = 0.8  # domyślnie szczęśliwy

        fake_face_data = {
            "has_face": True,
            "expressions": fake_expressions
        }

        # Aktualizuj promień głowy w rendererze
        self.face_renderer.set_head_radius(head_radius)

        # Renderuj twarz z fałszywymi danymi FaceMesh
        result_canvas = self.face_renderer.render_face(canvas, head_position, fake_face_data)

        self.performance.stop_timer()
        return result_canvas

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan integratora.
        """
        if hasattr(self, 'face_renderer'):
            self.face_renderer.reset()

        self.logger.debug(
            "StickFigureFaceIntegration",
            "Reset wewnętrznego stanu integratora twarzy",
            log_type="DRAWING"
        )
