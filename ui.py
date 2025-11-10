"""
UI and display module.

Handles all UI-related functions including debug overlays,
FPS display, and status messages.
"""

import cv2
import config


class FPSCounter:
    """
    Simple FPS counter that updates at specified intervals.
    """

    def __init__(self, update_interval=30):
        """
        Initialize FPS counter.

        Args:
            update_interval: Number of frames between FPS updates
        """
        self.counter = 0
        self.start_time = cv2.getTickCount()
        self.current_fps = 0
        self.update_interval = update_interval

    def update(self):
        """
        Update FPS counter. Should be called once per frame.

        Returns:
            float: Current FPS value
        """
        self.counter += 1

        if self.counter >= self.update_interval:
            end_time = cv2.getTickCount()
            time_diff = (end_time - self.start_time) / cv2.getTickFrequency()
            self.current_fps = self.update_interval / time_diff
            self.start_time = end_time
            self.counter = 0

        return self.current_fps

    def get_fps(self):
        """
        Get current FPS value without updating.

        Returns:
            float: Current FPS value
        """
        return self.current_fps


def draw_debug_info(canvas, fps, mouth_open, width, height):
    """
    Draw debug information overlay on canvas.

    Args:
        canvas: Canvas to draw on
        fps: Current FPS value
        mouth_open: Whether mouth is detected as open
        width: Canvas width
        height: Canvas height
    """
    text_color = config.DEBUG_TEXT_COLOR
    text_scale = config.DEBUG_TEXT_SCALE
    text_thickness = config.DEBUG_TEXT_THICKNESS

    # FPS display
    cv2.putText(
        canvas,
        f'FPS: {fps:.1f}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 255, 0),
        text_thickness
    )

    # Mouth status
    mouth_status = "OPEN" if mouth_open else "CLOSED"
    cv2.putText(
        canvas,
        f'Mouth: {mouth_status}',
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 255, 255),
        text_thickness
    )


def draw_no_person_message(canvas, width, height):
    """
    Draw 'no person detected' message on canvas.

    Args:
        canvas: Canvas to draw on
        width: Canvas width
        height: Canvas height
    """
    cv2.putText(
        canvas,
        config.MESSAGE_NO_PERSON,
        (width // 2 - 200, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        config.STICKFIGURE_COLOR,
        2
    )


def draw_debug_mode_indicator(canvas, height):
    """
    Draw debug mode indicator at bottom of canvas.

    Args:
        canvas: Canvas to draw on
        height: Canvas height
    """
    cv2.putText(
        canvas,
        config.MESSAGE_DEBUG_TOGGLE,
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 128, 128),
        1
    )


def draw_debug_camera_label(canvas, height):
    """
    Draw 'Debug Camera View' label on canvas.

    Args:
        canvas: Canvas to draw on
        height: Canvas height
    """
    cv2.putText(
        canvas,
        'Debug Camera View',
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        config.DEBUG_TEXT_COLOR,
        2
    )


def create_debug_overlay(frame, pose_results, face_results, landmarks,
                         width, height, mouth_open, fps,
                         stickfigure_drawer, face_drawer):
    """
    Create debug overlay with all detection visualizations.

    Args:
        frame: Original camera frame
        pose_results: MediaPipe pose detection results
        face_results: MediaPipe face mesh results
        landmarks: Pose landmarks
        width: Frame width
        height: Frame height
        mouth_open: Whether mouth is open
        fps: Current FPS
        stickfigure_drawer: Function to draw stickfigure
        face_drawer: Function to draw face landmarks

    Returns:
        numpy.ndarray: Frame with debug overlay
    """
    debug_canvas = frame.copy()

    # Draw stick figure overlay with debug markers
    if pose_results.pose_landmarks:
        stickfigure_drawer(
            debug_canvas,
            landmarks,
            width,
            height,
            mouth_open,
            draw_debug_markers=True
        )

    # Draw face landmarks
    if face_results.multi_face_landmarks:
        face_drawer(
            debug_canvas,
            face_results.multi_face_landmarks[0].landmark,
            width,
            height
        )

    # Add debug information
    draw_debug_info(debug_canvas, fps, mouth_open, width, height)
    draw_debug_camera_label(debug_canvas, height)

    return debug_canvas


def print_startup_info(width, height):
    """
    Print application startup information to console.

    Args:
        width: Camera width
        height: Camera height
    """
    print("Camera initialized")
    print(f"Resolution: {width}x{height}")
    print(config.MESSAGE_QUIT)
    print(config.MESSAGE_DEBUG_KEY)


def handle_key_press(key, debug_mode):
    """
    Handle keyboard input.

    Args:
        key: Key code from cv2.waitKey()
        debug_mode: Current debug mode state

    Returns:
        tuple: (should_quit: bool, new_debug_mode: bool)
    """
    should_quit = False
    new_debug_mode = debug_mode

    if key == ord('q'):
        should_quit = True
    elif key == ord('d'):
        new_debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if new_debug_mode else 'OFF'}")

        # Close debug window when turning off debug mode
        if not new_debug_mode:
            cv2.destroyWindow(config.WINDOW_NAME_DEBUG)

    return should_quit, new_debug_mode