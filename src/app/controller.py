#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application controller managing lifecycle and coordination."""

import signal
import time
from typing import Optional

from src.app.config_manager import ConfigurationManager
from src.app.ui_manager import UIManager
from src.app.video_pipeline import VideoPipeline
from src.utils.custom_logger import CustomLogger


class ApplicationController:
    """Main application controller coordinating all components.

    This class manages the application lifecycle, coordinates between
    video pipeline and UI, and handles user input events.
    """

    def __init__(
        self,
        config_manager: ConfigurationManager,
        logger: Optional[CustomLogger] = None
    ):
        """Initialize the application controller.

        Args:
            config_manager: Configuration manager instance
            logger: Optional custom logger
        """
        self.config = config_manager
        self.logger = logger or CustomLogger()

        # State flags
        self.running = False
        self.paused = False

        # Components (initialized in setup)
        self.pipeline: Optional[VideoPipeline] = None
        self.ui: Optional[UIManager] = None

        # Performance tracking
        self.frame_count = 0
        self.start_time = 0

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("Controller", "Application controller initialized")

    def setup(self) -> bool:
        """Setup all application components.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Create video pipeline
            self.pipeline = VideoPipeline(
                self.config.get_camera_config(),
                self.config.get_processing_config(),
                self.logger
            )

            if not self.pipeline.initialize():
                self.logger.error("Controller", "Failed to initialize video pipeline")
                return False

            # Create UI manager if preview enabled
            if self.config.get("app.show_preview", True):
                self.ui = UIManager(
                    self.config.get_ui_config(),
                    self.logger
                )

            self.logger.info("Controller", "All components setup successfully")
            return True

        except Exception as e:
            self.logger.error(
                "Controller",
                f"Failed to setup application: {str(e)}"
            )
            return False

    def run(self):
        """Run the main application loop."""
        if not self.setup():
            self.logger.error("Controller", "Setup failed, cannot start")
            return

        self.running = True
        self.start_time = time.time()

        self.logger.info("Controller", "Starting main loop")
        self._print_controls()

        try:
            while self.running:
                if not self.paused:
                    self._process_frame()
                else:
                    time.sleep(0.05)  # Reduce CPU when paused

                self._handle_input()
                self.frame_count += 1

        except KeyboardInterrupt:
            self.logger.info("Controller", "Interrupted by user")
        except Exception as e:
            self.logger.error(
                "Controller",
                f"Error in main loop: {str(e)}"
            )
        finally:
            self._cleanup()

    def _process_frame(self):
        """Process a single frame through the pipeline."""
        frame_data = self.pipeline.process_frame()

        if frame_data and self.ui:
            self.ui.update(frame_data)

    def _handle_input(self):
        """Handle keyboard input from UI."""
        if not self.ui:
            return

        key = self.ui.get_key_press()

        if key is None:
            return

        # Quit
        if key in ['q', 'ESC']:
            self.logger.info("Controller", "Quit requested by user")
            self.running = False

        # Pause/Resume
        elif key == 'p':
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            self.logger.info("Controller", f"Application {status}")

        # Debug toggle
        elif key == 'd':
            debug = not self.config.get("app.debug", False)
            self.config.set("app.debug", debug)
            self.logger.info("Controller", f"Debug mode: {debug}")

        # Flip camera
        elif key == 'f':
            flip = not self.config.get("camera.flip_horizontal", True)
            self.config.set("camera.flip_horizontal", flip)
            self.logger.info("Controller", f"Camera flip: {flip}")

        # Cycle mood
        elif key == 's':
            self._cycle_mood()

    def _cycle_mood(self):
        """Cycle through available moods."""
        moods = ["happy", "neutral", "sad", "surprised", "wink"]
        current = self.config.get("stick_figure.mood", "happy")

        try:
            idx = moods.index(current)
            new_mood = moods[(idx + 1) % len(moods)]
        except ValueError:
            new_mood = "neutral"

        self.config.set("stick_figure.mood", new_mood)
        self.logger.info("Controller", f"Changed mood to: {new_mood}")

    def _print_controls(self):
        """Print keyboard controls to console."""
        controls = [
            "=== Keyboard Controls ===",
            "q or ESC: Quit application",
            "p: Pause/Resume",
            "d: Toggle debug mode",
            "f: Flip camera horizontally",
            "s: Change stick figure mood",
            "========================="
        ]
        for line in controls:
            print(line)

    def _signal_handler(self, sig, frame):
        """Handle system signals for graceful shutdown.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        self.logger.info("Controller", f"Received signal {sig}, shutting down")
        self.running = False

    def _cleanup(self):
        """Cleanup all resources before exit."""
        self.logger.info("Controller", "Cleaning up resources")

        if self.pipeline:
            self.pipeline.shutdown()

        if self.ui:
            self.ui.cleanup()

        # Print statistics
        if self.start_time > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.logger.info(
                "Controller",
                f"Session stats: {self.frame_count} frames, "
                f"{elapsed:.1f}s, {avg_fps:.1f} FPS"
            )

        self.logger.info("Controller", "Cleanup complete")

    def stop(self):
        """Stop the application gracefully."""
        self.running = False
