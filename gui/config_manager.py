"""
Configuration manager for GUI application.

This module handles thread-safe configuration storage and updates
for real-time parameter changes in the GUI, with persistent storage
to disk using JSON files.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from persistent_config import PersistentConfig


@dataclass
class LiveConfig:
    """
    Thread-safe configuration that can be updated in real-time.

    This dataclass stores all configurable parameters and provides
    thread-safe access for concurrent updates from GUI and camera threads,
    with automatic persistence to disk.
    """

    # Appearance settings
    stickfigure_thickness: int = 4
    joint_radius: int = 6
    eye_radius_ratio: float = 0.12
    mouth_width_ratio: float = 0.5
    mouth_height_ratio: float = 0.25

    # Head proportions
    head_radius_ratio: float = 0.4
    neck_length_ratio: float = 0.6
    eye_y_offset_ratio: float = 0.25
    eye_spacing_ratio: float = 0.35
    mouth_y_offset_ratio: float = 0.4
    shoulder_curve_depth_ratio: float = 0.15

    # Detection sensitivity
    mouth_open_threshold_ratio: float = 0.025
    eyes_closed_ratio_threshold: float = 0.055
    eyes_closed_consecutive_frames: int = 3

    # MediaPipe settings
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5
    pose_model_complexity: int = 0

    # Virtual camera
    vcam_mirror_output: bool = True

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _persistent: Optional[PersistentConfig] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """Initialize persistent storage after dataclass initialization."""
        self._persistent = PersistentConfig(
            app_name="stickfigure_webcam",
            config_filename="config.json",
            debounce_seconds=2.0
        )
        self._load_from_disk()

    def _get_defaults(self):
        """
        Get default configuration values.

        Returns:
            dict: Dictionary of default configuration values.
        """
        return {
            'stickfigure_thickness': 4,
            'joint_radius': 6,
            'eye_radius_ratio': 0.12,
            'mouth_width_ratio': 0.5,
            'mouth_height_ratio': 0.25,
            'head_radius_ratio': 0.4,
            'neck_length_ratio': 0.6,
            'eye_y_offset_ratio': 0.25,
            'eye_spacing_ratio': 0.35,
            'mouth_y_offset_ratio': 0.4,
            'shoulder_curve_depth_ratio': 0.15,
            'mouth_open_threshold_ratio': 0.025,
            'eyes_closed_ratio_threshold': 0.055,
            'eyes_closed_consecutive_frames': 3,
            'pose_min_detection_confidence': 0.5,
            'pose_min_tracking_confidence': 0.5,
            'pose_model_complexity': 0,
            'vcam_mirror_output': True,
        }

    def _load_from_disk(self):
        """Load configuration from disk on startup."""
        defaults = self._get_defaults()
        loaded_config = self._persistent.load(defaults)

        # Apply loaded values (without triggering save)
        with self._lock:
            for key, value in loaded_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        print("[LiveConfig] Configuration loaded from disk")

    def _save_to_disk(self):
        """Save current configuration to disk (with debouncing)."""
        snapshot = self.get_snapshot()
        self._persistent.save(snapshot, debounce=True)

    def update(self, **kwargs):
        """
        Thread-safe update of configuration values with automatic persistence.

        Args:
            **kwargs: Configuration parameters to update.
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Save to disk (with debouncing)
        self._save_to_disk()

    def get_snapshot(self):
        """
        Get thread-safe snapshot of all configuration values.

        Returns:
            dict: Dictionary containing all current configuration values.
        """
        with self._lock:
            return {
                'stickfigure_thickness': self.stickfigure_thickness,
                'joint_radius': self.joint_radius,
                'eye_radius_ratio': self.eye_radius_ratio,
                'mouth_width_ratio': self.mouth_width_ratio,
                'mouth_height_ratio': self.mouth_height_ratio,
                'head_radius_ratio': self.head_radius_ratio,
                'neck_length_ratio': self.neck_length_ratio,
                'eye_y_offset_ratio': self.eye_y_offset_ratio,
                'eye_spacing_ratio': self.eye_spacing_ratio,
                'mouth_y_offset_ratio': self.mouth_y_offset_ratio,
                'shoulder_curve_depth_ratio': self.shoulder_curve_depth_ratio,
                'mouth_open_threshold_ratio': self.mouth_open_threshold_ratio,
                'eyes_closed_ratio_threshold': self.eyes_closed_ratio_threshold,
                'eyes_closed_consecutive_frames': self.eyes_closed_consecutive_frames,
                'pose_min_detection_confidence': self.pose_min_detection_confidence,
                'pose_min_tracking_confidence': self.pose_min_tracking_confidence,
                'pose_model_complexity': self.pose_model_complexity,
                'vcam_mirror_output': self.vcam_mirror_output,
            }

    def reset_to_defaults(self):
        """Reset all configuration values to their defaults and save immediately."""
        defaults = self._get_defaults()

        with self._lock:
            for key, value in defaults.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Force immediate save (no debouncing)
        if self._persistent:
            self._persistent.save(defaults, debounce=False)

        print("[LiveConfig] Configuration reset to defaults")

    def flush_to_disk(self):
        """Force immediate write of any pending changes to disk."""
        if self._persistent:
            self._persistent.flush()
            print("[LiveConfig] Configuration flushed to disk")
