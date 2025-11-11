"""
Rendering management for main window.

This module handles frame rendering and virtual camera output.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

import config
from stickfigure import draw_stickfigure
from ui import draw_no_person_message
from virtual_camera import VirtualCameraOutput


class RenderingManager(QObject):
    """
    Rendering manager for main window.

    Handles stickfigure rendering and virtual camera output.
    """

    def __init__(self, live_config, ui_manager):
        """
        Initialize rendering manager.

        Args:
            live_config: LiveConfig instance
            ui_manager: MainWindowUI instance
        """
        super().__init__()

        self.live_config = live_config
        self.ui_manager = ui_manager

        self.current_frame_data = None
        self.vcam = None

        self._init_virtual_camera()

        print("[RenderingManager] Initialized")

    def _init_virtual_camera(self):
        """Initialize virtual camera output."""
        try:
            self.vcam = VirtualCameraOutput(
                config.CAMERA_WIDTH,
                config.CAMERA_HEIGHT,
                fps=config.CAMERA_FPS,
                mirror=self.live_config.vcam_mirror_output
            )

            if self.vcam.start():
                self.ui_manager.update_vcam_status(
                    f"Virtual Camera: Active ({self.vcam.camera.device})",
                    "#00ff00"
                )
                print("[RenderingManager] Virtual camera started")
            else:
                self.ui_manager.update_vcam_status(
                    "Virtual Camera: Failed to Start",
                    "#ff0000"
                )
        except Exception as e:
            print(f"[RenderingManager] Virtual camera error: {e}")
            self.ui_manager.update_vcam_status(
                "Virtual Camera: Error",
                "#ff0000"
            )

    @pyqtSlot(np.ndarray, object, object, bool, bool)
    def on_frame_ready(self, frame, pose_results, face_results, mouth_open, eyes_closed):
        """Handle new frame ready for rendering."""
        # Store frame data
        self.current_frame_data = {
            'frame': frame,
            'pose_results': pose_results,
            'face_results': face_results,
            'mouth_open': mouth_open,
            'eyes_closed': eyes_closed
        }

        # Render stickfigure
        self._render_stickfigure()

        # Update debug window if visible
        debug_window = self.ui_manager.get_debug_window()
        if debug_window and debug_window.isVisible():
            debug_window.update_frame_data(self.current_frame_data)

    def _render_stickfigure(self):
        """Render stickfigure with current configuration."""
        if not self.current_frame_data:
            return

        data = self.current_frame_data
        frame = data['frame']
        height, width = frame.shape[:2]

        # Create canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Get current config snapshot
        cfg = self.live_config.get_snapshot()

        # Temporarily update config module for drawing functions
        self._apply_config_to_globals(cfg)

        # Draw stickfigure or message
        if data['pose_results'].pose_landmarks:
            draw_stickfigure(
                canvas,
                data['pose_results'].pose_landmarks.landmark,
                width,
                height,
                data['mouth_open'],
                data['eyes_closed'],
                draw_debug_markers=False
            )
        else:
            draw_no_person_message(canvas, width, height)

        # Display in UI
        self._display_frame(canvas)

        # Send to virtual camera
        self._send_to_virtual_camera(canvas)

    def _apply_config_to_globals(self, cfg):
        """Apply configuration to global config module."""
        config.STICKFIGURE_THICKNESS = cfg['stickfigure_thickness']
        config.JOINT_RADIUS = cfg['joint_radius']
        config.EYE_RADIUS_RATIO = cfg['eye_radius_ratio']
        config.MOUTH_WIDTH_RATIO = cfg['mouth_width_ratio']
        config.MOUTH_HEIGHT_RATIO = cfg['mouth_height_ratio']
        config.HEAD_RADIUS_RATIO = cfg['head_radius_ratio']
        config.NECK_LENGTH_RATIO = cfg['neck_length_ratio']
        config.EYE_Y_OFFSET_RATIO = cfg['eye_y_offset_ratio']
        config.EYE_SPACING_RATIO = cfg['eye_spacing_ratio']
        config.MOUTH_Y_OFFSET_RATIO = cfg['mouth_y_offset_ratio']
        config.SHOULDER_CURVE_DEPTH_RATIO = cfg['shoulder_curve_depth_ratio']

    def _display_frame(self, canvas):
        """Display frame in video widget."""
        if not self.ui_manager.video_widget:
            return

        # Convert to QImage
        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.ui_manager.video_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.ui_manager.video_widget.setPixmap(scaled_pixmap)

    def _send_to_virtual_camera(self, canvas):
        """Send frame to virtual camera."""
        if self.vcam and self.vcam.is_active:
            try:
                self.vcam.send_frame(canvas)
            except Exception as e:
                print(f"[RenderingManager] Virtual camera send error: {e}")

    @pyqtSlot(dict)
    def on_config_changed(self, changes):
        """Handle configuration change."""
        # Re-render current frame with new settings
        if self.current_frame_data:
            self._render_stickfigure()

    @pyqtSlot(bool)
    def on_mirror_changed(self, mirror):
        """Handle mirror setting change."""
        if self.vcam:
            self.vcam.set_mirror(mirror)

    def cleanup(self):
        """Clean up resources."""
        if self.vcam and self.vcam.is_active:
            self.vcam.stop()
        print("[RenderingManager] Cleanup complete")
