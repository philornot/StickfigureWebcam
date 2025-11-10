"""
Virtual camera output module.

This module handles sending frames to a virtual camera device
using pyvirtualcam, making the stickfigure output available
to applications like Zoom, Teams, etc.
"""

import numpy as np
import pyvirtualcam


class VirtualCameraOutput:
    """
    Wrapper for pyvirtualcam that sends frames to virtual camera.
    """

    def __init__(self, width, height, fps=30):
        """
        Initialize virtual camera output.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frame rate (default: 30)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.is_active = False

    def start(self):
        """
        Start the virtual camera.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.RGB
            )
            self.is_active = True
            print(f"Virtual camera started: {self.camera.device}")
            return True
        except Exception as e:
            print(f"Error starting virtual camera: {e}")
            print("Make sure you have installed:")
            print("  - Windows/Mac: OBS Studio")
            print("  - Linux: v4l2loopback")
            return False

    def send_frame(self, frame):
        """
        Send a frame to the virtual camera.

        Args:
            frame: BGR frame from OpenCV (numpy array)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_active or self.camera is None:
            return False

        try:
            # Convert BGR to RGB (OpenCV uses BGR, pyvirtualcam expects RGB)
            frame_rgb = frame[:, :, ::-1]

            # Ensure correct size
            if frame_rgb.shape[:2] != (self.height, self.width):
                import cv2
                frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))

            # Send frame
            self.camera.send(frame_rgb)
            return True

        except Exception as e:
            print(f"Error sending frame to virtual camera: {e}")
            return False

    def sleep_until_next_frame(self):
        """
        Adaptively sleep until next frame is due.
        This helps maintain target FPS.
        """
        if self.camera:
            self.camera.sleep_until_next_frame()

    def stop(self):
        """
        Stop the virtual camera and release resources.
        """
        if self.camera:
            try:
                self.camera.close()
                print("Virtual camera stopped")
            except:
                pass
            finally:
                self.camera = None
                self.is_active = False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Example usage function for testing
def test_virtual_camera():
    """
    Test function to verify virtual camera works.
    Creates a simple animated pattern.
    """
    import cv2

    width, height = 640, 480

    with VirtualCameraOutput(width, height, fps=30) as vcam:
        if not vcam.is_active:
            print("Failed to start virtual camera")
            return

        print("Virtual camera test running...")
        print("Check your video apps (Zoom, Teams, etc.) for 'OBS Virtual Camera'")
        print("Press Ctrl+C to stop")

        try:
            frame_count = 0
            while True:
                # Create test pattern
                frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Moving circle
                x = int(width / 2 + width / 3 * np.cos(frame_count * 0.05))
                y = int(height / 2 + height / 3 * np.sin(frame_count * 0.05))
                cv2.circle(frame, (x, y), 50, (0, 255, 0), -1)

                # Frame counter
                cv2.putText(
                    frame,
                    f'Frame: {frame_count}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                # Send to virtual camera
                vcam.send_frame(frame)
                vcam.sleep_until_next_frame()

                frame_count += 1

        except KeyboardInterrupt:
            print("\nTest stopped")


if __name__ == "__main__":
    test_virtual_camera()
