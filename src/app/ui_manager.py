#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI manager for preview windows and debug overlays."""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger


class UIManager:
    """Manages preview windows and debug information display."""

    def __init__(self, ui_config: Dict[str, Any], logger: Optional[CustomLogger] = None):
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
        if not self.show_preview or not frame_data:
            return

        original = frame_data.get("original_frame")
        processed = frame_data.get("processed_frame")
        fps = frame_data.get("fps", 0.0)

        if original is not None:
            display_frame = self._add_overlay(original.copy(), frame_data)
            cv2.imshow("Camera Preview", display_frame)

        if processed is not None:
            output_frame = self._add_output_overlay(processed, fps)
            cv2.imshow("Stick Figure Output", output_frame)

    def _add_overlay(self, frame: np.ndarray, frame_data: Dict[str, Any]) -> np.ndarray:
        """Add debug overlay to frame.

        Args:
            frame: Original frame
            frame_data: Frame processing data

        Returns:
            Frame with overlay
        """
        display = frame.copy()
        fps = frame_data.get("fps", 0.0)

        # FPS counter (always visible)
        cv2.putText(
            display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        # Debug mode indicator
        if self.show_debug:
            cv2.putText(
                display, "DEBUG MODE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        # Draw landmarks in debug mode
        if self.show_debug or self.show_landmarks:
            display = self._draw_debug_overlays(display, frame_data)

        # Status info
        if self.show_debug:
            display = self._draw_status_info(display, frame_data)

        return display

    def _draw_debug_overlays(self, frame: np.ndarray, frame_data: Dict[str, Any]) -> np.ndarray:
        """Draw all debug visualizations on frame.

        Args:
            frame: Input frame
            frame_data: Frame data containing detection results

        Returns:
            Frame with debug overlays
        """
        if frame_data is None:
            return frame

        face_data = frame_data.get("face_data", {})
        h, w = frame.shape[:2]

        # Draw face mesh if available
        if face_data.get("has_face", False) and face_data.get("landmarks"):
            landmarks = face_data["landmarks"]
            frame = self._draw_face_mesh(frame, landmarks)

        # Draw upper body skeleton if available
        if "upper_body_data" in frame_data and frame_data["upper_body_data"] is not None:
            upper_body = frame_data["upper_body_data"]
            frame = self._draw_upper_body_skeleton(frame, upper_body)

        # Draw hand landmarks if available
        if face_data.get("hands_data"):
            hands_data = face_data["hands_data"]
            frame = self._draw_hand_landmarks(frame, hands_data, w, h)

        return frame

    def _draw_face_mesh(self, frame: np.ndarray, landmarks: list) -> np.ndarray:
        """Draw face mesh landmarks on frame.

        Args:
            frame: Input frame
            landmarks: List of facial landmarks (x, y, z, visibility)

        Returns:
            Frame with face mesh drawn
        """
        h, w = frame.shape[:2]

        # Draw all landmarks as small dots
        for i, landmark in enumerate(landmarks):
            x, y, _, _ = landmark
            cx, cy = int(x * w), int(y * h)

            # Color coding for different face regions
            if i < 10:  # Face oval
                color = (0, 255, 0)  # Green
            elif 10 <= i < 68:  # Eyebrows and eyes
                color = (255, 0, 0)  # Blue
            elif 68 <= i < 200:  # Nose and mouth
                color = (0, 0, 255)  # Red
            else:  # Rest
                color = (255, 255, 0)  # Cyan

            cv2.circle(frame, (cx, cy), 1, color, -1)

        # Draw key face contours
        # Face oval
        face_oval = [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
            10,
        ]

        for i in range(len(face_oval) - 1):
            idx1, idx2 = face_oval[i], face_oval[i + 1]
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                pt1 = (int(landmarks[idx1][0] * w), int(landmarks[idx1][1] * h))
                pt2 = (int(landmarks[idx2][0] * w), int(landmarks[idx2][1] * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)

        # Draw eyes contours
        left_eye = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 33]
        right_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 362]

        for eye_contour in [left_eye, right_eye]:
            for i in range(len(eye_contour) - 1):
                idx1, idx2 = eye_contour[i], eye_contour[i + 1]
                if idx1 < len(landmarks) and idx2 < len(landmarks):
                    pt1 = (int(landmarks[idx1][0] * w), int(landmarks[idx1][1] * h))
                    pt2 = (int(landmarks[idx2][0] * w), int(landmarks[idx2][1] * h))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

        # Draw mouth contour
        mouth = [
            61,
            185,
            40,
            39,
            37,
            0,
            267,
            269,
            270,
            409,
            291,
            375,
            321,
            405,
            314,
            17,
            84,
            181,
            91,
            146,
            61,
        ]

        for i in range(len(mouth) - 1):
            idx1, idx2 = mouth[i], mouth[i + 1]
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                pt1 = (int(landmarks[idx1][0] * w), int(landmarks[idx1][1] * h))
                pt2 = (int(landmarks[idx2][0] * w), int(landmarks[idx2][1] * h))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 1)

        return frame

    def _draw_upper_body_skeleton(
        self, frame: np.ndarray, upper_body_data: Dict[str, Any]
    ) -> np.ndarray:
        """Draw upper body skeleton (shoulders, elbows, wrists).

        Args:
            frame: Input frame
            upper_body_data: Upper body detection data

        Returns:
            Frame with skeleton drawn
        """
        # Draw shoulders line
        if upper_body_data.get("shoulder_positions"):
            shoulders = upper_body_data["shoulder_positions"]
            if shoulders[0] and shoulders[1]:
                cv2.line(frame, shoulders[0], shoulders[1], (255, 255, 0), 3)
                cv2.circle(frame, shoulders[0], 5, (255, 0, 255), -1)
                cv2.circle(frame, shoulders[1], 5, (255, 0, 255), -1)

        # Draw arms (shoulder -> elbow -> wrist)
        if upper_body_data.get("elbow_positions"):
            elbows = upper_body_data["elbow_positions"]
            shoulders = upper_body_data.get("shoulder_positions", (None, None))

            # Left arm
            if shoulders[0] and elbows[0]:
                cv2.line(frame, shoulders[0], elbows[0], (0, 255, 255), 2)
                cv2.circle(frame, elbows[0], 5, (0, 255, 0), -1)

            # Right arm
            if shoulders[1] and elbows[1]:
                cv2.line(frame, shoulders[1], elbows[1], (0, 255, 255), 2)
                cv2.circle(frame, elbows[1], 5, (0, 255, 0), -1)

        # Draw forearms (elbow -> wrist)
        if upper_body_data.get("wrist_positions"):
            wrists = upper_body_data["wrist_positions"]
            elbows = upper_body_data.get("elbow_positions", (None, None))

            # Left forearm
            if elbows[0] and wrists[0]:
                cv2.line(frame, elbows[0], wrists[0], (255, 0, 255), 2)
                cv2.circle(frame, wrists[0], 5, (0, 0, 255), -1)

            # Right forearm
            if elbows[1] and wrists[1]:
                cv2.line(frame, elbows[1], wrists[1], (255, 0, 255), 2)
                cv2.circle(frame, wrists[1], 5, (0, 0, 255), -1)

        return frame

    def _draw_hand_landmarks(
        self, frame: np.ndarray, hands_data: Dict[str, Any], w: int, h: int
    ) -> np.ndarray:
        """Draw hand landmarks if detected.

        Args:
            frame: Input frame
            hands_data: Hand detection data
            w: Frame width
            h: Frame height

        Returns:
            Frame with hand landmarks
        """
        # Left hand
        if hands_data.get("left_hand"):
            left_hand = hands_data["left_hand"]
            if "wrist" in left_hand:
                wrist = left_hand["wrist"]
                wx = int(wrist[0] * w if wrist[0] <= 1.0 else wrist[0])
                wy = int(wrist[1] * h if wrist[1] <= 1.0 else wrist[1])
                cv2.circle(frame, (wx, wy), 8, (255, 255, 0), -1)
                cv2.putText(
                    frame, "L", (wx - 15, wy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )

        # Right hand
        if hands_data.get("right_hand"):
            right_hand = hands_data["right_hand"]
            if "wrist" in right_hand:
                wrist = right_hand["wrist"]
                wx = int(wrist[0] * w if wrist[0] <= 1.0 else wrist[0])
                wy = int(wrist[1] * h if wrist[1] <= 1.0 else wrist[1])
                cv2.circle(frame, (wx, wy), 8, (0, 255, 255), -1)
                cv2.putText(
                    frame, "R", (wx - 15, wy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

        return frame

    def _draw_status_info(self, frame: np.ndarray, frame_data: Dict[str, Any]) -> np.ndarray:
        """Draw detailed status information.

        Args:
            frame: Input frame
            frame_data: Frame data

        Returns:
            Frame with status info
        """
        face_data = frame_data.get("face_data", {})
        y_offset = 100

        # Face detection status
        has_face = face_data.get("has_face", False)
        status_text = "Face: DETECTED" if has_face else "Face: NOT DETECTED"
        color = (0, 255, 0) if has_face else (0, 0, 255)
        cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

        # Expression values
        if has_face and "expressions" in face_data:
            expressions = face_data["expressions"]

            cv2.putText(
                frame,
                f"Smile: {expressions.get('smile', 0.5):.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )
            y_offset += 25

            cv2.putText(
                frame,
                f"Mouth open: {expressions.get('mouth_open', 0.0):.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )
            y_offset += 25

        # Upper body detection status
        if "upper_body_data" in frame_data:
            upper_body = frame_data["upper_body_data"]

            shoulders_status = (
                "Shoulders: OK" if upper_body.get("has_shoulders") else "Shoulders: NO"
            )
            color = (0, 255, 0) if upper_body.get("has_shoulders") else (0, 0, 255)
            cv2.putText(
                frame, shoulders_status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )
            y_offset += 25

            arms_status = "Arms: OK" if upper_body.get("has_arms") else "Arms: NO"
            color = (0, 255, 0) if upper_body.get("has_arms") else (0, 0, 255)
            cv2.putText(frame, arms_status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 25

        # Add legend at the bottom
        legend_y = frame.shape[0] - 100
        cv2.putText(
            frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        legend_y += 20
        cv2.putText(
            frame,
            "Green=Face | Blue=Eyes | Red=Mouth | Cyan=Body",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        legend_y += 18
        cv2.putText(
            frame,
            "Yellow=Shoulders | Magenta=Elbows | Red=Wrists",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return frame

    def _add_output_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
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
            display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1
        )

        return display

    def toggle_debug(self):
        """Toggle debug mode."""
        self.show_debug = not self.show_debug
        self.logger.info("UI", f"Debug mode: {'ON' if self.show_debug else 'OFF'}")

    def toggle_landmarks(self):
        """Toggle landmarks display."""
        self.show_landmarks = not self.show_landmarks
        self.logger.info("UI", f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")

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
            27: "ESC",
            ord("q"): "q",
            ord("p"): "p",
            ord("d"): "d",
            ord("f"): "f",
            ord("s"): "s",
            ord("l"): "l",  # Toggle landmarks
        }

        return key_map.get(key)

    def cleanup(self):
        """Cleanup UI resources."""
        cv2.destroyAllWindows()
        self.logger.info("UI", "UI cleanup complete")
