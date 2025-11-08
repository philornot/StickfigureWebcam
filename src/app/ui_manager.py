#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI manager for preview windows and debug overlays."""

from typing import Dict, Any, Optional

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger


class UIManager:
    """Manages preview windows and debug information display."""

    def __init__(
        self,
        ui_config: Dict[str, Any],
        logger: Optional[CustomLogger] = None
    ):
        """Initialize the UI manager.

        Args:
            ui_config: UI configuration dict
            logger: Optional custom logger
        """
        self.config = ui_config
        self.logger = logger or CustomLogger()

        self.show_preview = ui_config.get("show_preview", True)
        self.show_debug = ui_config.get("show_debug", False)
        self.show_landmarks = ui_config.get("show_landmarks", False)

        self.logger.info("UI", "UI Manager initialized")

    def update(self, frame_data: Dict[str, Any]):
        """Update UI with new frame data.

        Args:
            frame_data: Dictionary containing frame information
        """
        if not self.show_preview:
            return

        original = frame_data.get("original_frame")
        processed = frame_data.get("processed_frame")
        fps = frame_data.get("fps", 0.0)

        if original is not None:
            display_frame = self._add_overlay(original, frame_data)
            cv2.imshow("Camera Preview", display_frame)

        if processed is not None:
            output_frame = self._add_output_overlay(processed, fps)
            cv2.imshow("Stick Figure Output", output_frame)

    def _add_overlay(
        self,
        frame: np.ndarray,
        frame_data: Dict[str, Any]
    ) -> np.ndarray:
        """Add debug overlay to frame.

        Args:
            frame: Original frame
            frame_data: Frame processing data

        Returns:
            Frame with overlay
        """
        display = frame.copy()
        fps = frame_data.get("fps", 0.0)

        # FPS counter
        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # Debug info
        if self.show_debug:
            face_data = frame_data.get("face_data", {})
            has_face = face_data.get("has_face", False)

            status_text = "Face: DETECTED" if has_face else "Face: NOT DETECTED"
            color = (0, 255, 0) if has_face else (0, 0, 255)

            cv2.putText(
                display,
                status_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Expression values
            if has_face and "expressions" in face_data:
                expressions = face_data["expressions"]
                smile = expressions.get("smile", 0.5)
                mouth_open = expressions.get("mouth_open", 0.0)

                cv2.putText(
                    display,
                    f"Smile: {smile:.2f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )

                cv2.putText(
                    display,
                    f"Mouth: {mouth_open:.2f}",
                    (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )

        return display

    def _add_output_overlay(
        self,
        frame: np.ndarray,
        fps: float
    ) -> np.ndarray:
        """Add overlay to output frame.

        Args:
            frame: Stick figure frame
            fps: Current FPS

        Returns:
            Frame with overlay
        """
        display = frame.copy()

        # FPS counter
        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 100),
            1
        )

        return display

    def get_key_press(self, wait_ms: int = 1) -> Optional[str]:
        """Get keyboard input from OpenCV windows.

        Args:
            wait_ms: Milliseconds to wait for key press

        Returns:
            Key name or None
        """
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 255:  # No key pressed
            return None

        # Map key codes to names
        key_map = {
            27: 'ESC',
            ord('q'): 'q',
            ord('p'): 'p',
            ord('d'): 'd',
            ord('f'): 'f',
            ord('s'): 's',
        }

        return key_map.get(key)

    def cleanup(self):
        """Cleanup UI resources."""
        cv2.destroyAllWindows()
        self.logger.info("UI", "UI cleanup complete")
