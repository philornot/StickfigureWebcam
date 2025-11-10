"""
Main application module for Stickfigure Webcam.

This module contains the main application loop that ties together
all components: camera, detection, drawing, UI, and virtual camera output.
"""

import cv2

import config
from camera_manager import CameraManager
from detection_worker import DetectionWorker
from renderer import Renderer
from ui import FPSCounter, handle_key_press, print_startup_info
from virtual_camera import VirtualCameraOutput


class StickfigureWebcam:
    """
    Main application class for Stickfigure Webcam.

    This refactored version delegates responsibilities to specialized
    worker classes for better code organization.
    """

    def __init__(self, camera_id=None):
        """
        Initialize the application.

        Args:
            camera_id: Optional camera ID. If None, will show selection dialog.
        """
        print("[StickfigureWebcam] Initializing...")

        # Initialize camera
        self.camera_manager = CameraManager(camera_id=camera_id, auto_select=True)
        width, height = self.camera_manager.get_resolution()

        # Initialize detection worker
        self.detection_worker = DetectionWorker(self.camera_manager.get_capture())

        # Initialize renderer
        self.renderer = Renderer(width, height)

        # Initialize FPS counter
        self.fps_counter = FPSCounter(config.FPS_UPDATE_INTERVAL)

        # UI state
        self.debug_mode = False

        # Initialize virtual camera
        self.vcam = VirtualCameraOutput(
            width,
            height,
            fps=config.CAMERA_FPS,
            mirror=config.VCAM_MIRROR_OUTPUT
        )
        self.vcam.start()

        # Print startup info
        print_startup_info(width, height)
        print(f"Using camera: {self.camera_manager.camera_id}")
        print("[StickfigureWebcam] Initialization complete")

    def run(self):
        """
        Main application loop.

        Starts worker threads and enters the main rendering loop,
        handling user input and display updates.
        """
        print("[StickfigureWebcam] Starting main loop...")

        # Start worker threads
        self.detection_worker.start()

        try:
            self._main_loop()
        finally:
            print("[StickfigureWebcam] Main loop exited")

    def _main_loop(self):
        """
        Main rendering and event loop.

        Continuously processes detection results, renders views,
        and handles user input until quit is requested.
        """
        while True:
            # Get detection results from worker
            detection_data = self.detection_worker.get_detection_result(timeout=0.1)

            if detection_data is None:
                # No new data, continue waiting
                continue

            # Update FPS
            self.fps_counter.update()
            current_fps = self.fps_counter.get_fps()

            # Render views
            stickfigure_canvas, debug_canvas = self.renderer.render_both_views(
                detection_data,
                current_fps,
                debug_mode=self.debug_mode
            )

            # Display main view
            cv2.imshow(config.WINDOW_NAME_STICKFIGURE, stickfigure_canvas)

            # Send to virtual camera
            if self.vcam.is_active:
                self.vcam.send_frame(stickfigure_canvas)
                self.vcam.sleep_until_next_frame()

            # Display debug view if enabled
            if self.debug_mode and debug_canvas is not None:
                cv2.imshow(config.WINDOW_NAME_DEBUG, debug_canvas)

            # Handle keyboard input
            if not self._handle_input():
                break

    def _handle_input(self):
        """
        Handle keyboard input.

        Returns:
            bool: False if quit is requested, True otherwise.
        """
        key = cv2.waitKey(1) & 0xFF

        # Handle standard keys (quit, debug toggle)
        should_quit, self.debug_mode = handle_key_press(key, self.debug_mode)

        if should_quit:
            return False

        # Handle mirror toggle
        if key == ord('m'):
            self.vcam.set_mirror(not self.vcam.mirror)

        return True

    def cleanup(self):
        """Clean up all resources."""
        print("[StickfigureWebcam] Cleaning up...")

        # Stop worker threads
        self.detection_worker.stop()

        # Clean up MediaPipe resources
        self.detection_worker.cleanup()

        # Stop virtual camera
        if self.vcam.is_active:
            self.vcam.stop()

        # Release camera
        self.camera_manager.release()

        # Close all windows
        cv2.destroyAllWindows()

        print("[StickfigureWebcam] Cleanup complete")


def main():
    """Application entry point."""
    print("=" * 60)
    print("STICKFIGURE WEBCAM")
    print("=" * 60)

    app = None
    try:
        app = StickfigureWebcam()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if app is not None:
            app.cleanup()


if __name__ == "__main__":
    main()
