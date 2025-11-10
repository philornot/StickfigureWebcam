"""
Stick figure drawing module.

This module handles all rendering of the stick figure based on pose landmarks,
including body proportions, facial features, and debug visualization.
"""

import cv2
import numpy as np

import config


def get_point(landmarks, idx, width, height):
    """
    Convert normalized landmark to pixel coordinates.

    Args:
        landmarks: MediaPipe pose landmarks
        idx: Landmark index
        width: Canvas width
        height: Canvas height

    Returns:
        tuple: (x, y) pixel coordinates
    """
    lm = landmarks[idx]
    x = int(lm.x * width)
    y = int(lm.y * height)
    return (x, y)


def draw_head_and_face(canvas, head_center, head_radius, mouth_open, eyes_closed):
    """
    Draw the head, eyes, and mouth of the stick figure.

    Args:
        canvas: Canvas to draw on
        head_center: (x, y) center position of the head
        head_radius: Radius of the head circle
        mouth_open: Whether mouth is open
        eyes_closed: Whether eyes are closed
    """
    color = config.STICKFIGURE_COLOR
    thickness = config.STICKFIGURE_THICKNESS

    # Draw head circle
    cv2.circle(canvas, head_center, head_radius, color, thickness)

    # EYES
    eye_y_offset = int(-head_radius * config.EYE_Y_OFFSET_RATIO)
    eye_spacing = int(head_radius * config.EYE_SPACING_RATIO)
    eye_radius = max(3, int(head_radius * config.EYE_RADIUS_RATIO))

    left_eye_center = (head_center[0] - eye_spacing, head_center[1] + eye_y_offset)
    right_eye_center = (head_center[0] + eye_spacing, head_center[1] + eye_y_offset)

    if eyes_closed:
        # Draw closed eyes (lines)
        line_len = int(eye_radius * 1.5)
        cv2.line(canvas,
                 (left_eye_center[0] - line_len, left_eye_center[1]),
                 (left_eye_center[0] + line_len, left_eye_center[1]),
                 color, 2)
        cv2.line(canvas,
                 (right_eye_center[0] - line_len, right_eye_center[1]),
                 (right_eye_center[0] + line_len, right_eye_center[1]),
                 color, 2)
    else:
        # Draw open eyes (circles)
        cv2.circle(canvas, left_eye_center, eye_radius, color, -1)
        cv2.circle(canvas, right_eye_center, eye_radius, color, -1)

    # MOUTH
    mouth_y_offset = int(head_radius * config.MOUTH_Y_OFFSET_RATIO)
    mouth_center = (head_center[0], head_center[1] + mouth_y_offset)
    mouth_width = int(head_radius * config.MOUTH_WIDTH_RATIO)

    if mouth_open:
        # Open mouth - draw as a circle/oval
        mouth_height = int(head_radius * config.MOUTH_HEIGHT_RATIO)
        cv2.ellipse(canvas, mouth_center, (mouth_width // 2, mouth_height // 2),
                    0, 0, 360, color, -1)
    else:
        # Closed mouth - draw as a line
        cv2.line(canvas,
                 (mouth_center[0] - mouth_width // 2, mouth_center[1]),
                 (mouth_center[0] + mouth_width // 2, mouth_center[1]),
                 color, 2)


def draw_curved_shoulders(canvas, left_shoulder, right_shoulder, shoulder_center, shoulder_width):
    """Draw curved shoulder line using vectorized NumPy operations.

    This function uses quadratic Bezier curve with fully vectorized
    NumPy calculations for better performance.

    Args:
        canvas: Canvas to draw on.
        left_shoulder: (x, y) position of left shoulder.
        right_shoulder: (x, y) position of right shoulder.
        shoulder_center: (x, y) center position between shoulders.
        shoulder_width: Distance between shoulders.
    """
    color = config.STICKFIGURE_COLOR
    thickness = config.STICKFIGURE_THICKNESS

    # Calculate control point for the curve
    shoulder_curve_depth = int(shoulder_width * config.SHOULDER_CURVE_DEPTH_RATIO)
    control_point = (shoulder_center[0], shoulder_center[1] + shoulder_curve_depth)

    # Vectorized Bezier curve calculation
    num_points = 20
    t = np.linspace(0, 1, num_points + 1)

    # Quadratic Bezier formula using NumPy broadcasting
    # B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    one_minus_t = 1 - t
    term1 = one_minus_t ** 2
    term2 = 2 * one_minus_t * t
    term3 = t ** 2

    x_coords = (term1 * left_shoulder[0] +
                term2 * control_point[0] +
                term3 * right_shoulder[0]).astype(np.int32)

    y_coords = (term1 * left_shoulder[1] +
                term2 * control_point[1] +
                term3 * right_shoulder[1]).astype(np.int32)

    # Stack coordinates into shape (N, 1, 2) as required by cv2.polylines
    shoulder_curve_points = np.column_stack([x_coords, y_coords]).reshape((-1, 1, 2))

    # Draw the curved shoulder line
    cv2.polylines(canvas, [shoulder_curve_points], False, color, thickness)


def draw_pose_debug_markers(canvas, landmarks, width, height):
    """Draw debug markers showing all detected landmark positions.

    Uses pre-allocated arrays and batch operations for better performance.

    Args:
        canvas: Canvas to draw on.
        landmarks: MediaPipe pose landmarks.
        width: Canvas width.
        height: Canvas height.
    """
    debug_colors = config.DEBUG_MARKER_COLORS

    # Pre-allocate arrays for batch processing
    points = []
    colors = []
    indices = []

    # Collect all valid points first
    for idx in range(33):
        try:
            lm = landmarks[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)

            if idx == 0:  # Nose
                marker_color = debug_colors['nose']
            elif idx in [11, 12, 13, 14, 15, 16]:  # Arms
                marker_color = debug_colors['arms']
            elif idx in [23, 24, 25, 26, 27, 28]:  # Legs
                marker_color = debug_colors['legs']
            else:  # Other points
                marker_color = debug_colors['other']

            points.append((x, y))
            colors.append(marker_color)
            indices.append(idx)
        except:
            pass

    # Draw all markers
    for point, color, idx in zip(points, colors, indices):
        cv2.circle(canvas, point, 3, color, -1)
        cv2.putText(canvas, str(idx),
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    color, 1)


def draw_stickfigure(canvas, landmarks, width, height, mouth_open=False,
                     eyes_closed=False, draw_debug_markers=False):
    """Draw a stick figure based on pose landmarks with natural proportions.

    Args:
        canvas: Numpy array to draw on.
        landmarks: MediaPipe pose landmarks.
        width: Canvas width.
        height: Canvas height.
        mouth_open: Whether mouth is open.
        eyes_closed: Whether eyes are closed.
        draw_debug_markers: Whether to draw small markers at detected landmark positions.
    """
    if not landmarks:
        return

    color = config.STICKFIGURE_COLOR
    thickness = config.STICKFIGURE_THICKNESS
    joint_radius = config.JOINT_RADIUS

    try:
        # Optimized point extraction - convert once and store
        def get_point_fast(idx):
            lm = landmarks[idx]
            return (int(lm.x * width), int(lm.y * height))

        # Extract all points at once
        left_shoulder = get_point_fast(11)
        right_shoulder = get_point_fast(12)
        left_elbow = get_point_fast(13)
        right_elbow = get_point_fast(14)
        left_wrist = get_point_fast(15)
        right_wrist = get_point_fast(16)
        left_hip = get_point_fast(23)
        right_hip = get_point_fast(24)
        left_knee = get_point_fast(25)
        right_knee = get_point_fast(26)
        left_ankle = get_point_fast(27)
        right_ankle = get_point_fast(28)

        # Calculate centers
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2
        )
        hip_center = (
            (left_hip[0] + right_hip[0]) // 2,
            (left_hip[1] + right_hip[1]) // 2
        )

        # Calculate shoulder width for proportional head sizing
        shoulder_width = np.sqrt(
            (right_shoulder[0] - left_shoulder[0]) ** 2 +
            (right_shoulder[1] - left_shoulder[1]) ** 2
        )

        # HEAD
        head_radius = int(shoulder_width * config.HEAD_RADIUS_RATIO)
        head_radius = max(config.HEAD_RADIUS_MIN,
                          min(head_radius, config.HEAD_RADIUS_MAX))
        neck_length = int(head_radius * config.NECK_LENGTH_RATIO)
        head_center = (shoulder_center[0], shoulder_center[1] - head_radius - neck_length)

        # Draw head and face
        draw_head_and_face(canvas, head_center, head_radius, mouth_open, eyes_closed)

        # NECK
        neck_top = (head_center[0], head_center[1] + head_radius)
        cv2.line(canvas, neck_top, shoulder_center, color, thickness)

        # TORSO
        draw_curved_shoulders(canvas, left_shoulder, right_shoulder, shoulder_center, shoulder_width)
        cv2.line(canvas, shoulder_center, hip_center, color, thickness)
        cv2.line(canvas, left_hip, right_hip, color, thickness)

        # Batch line drawing for limbs - reduces function call overhead
        limb_pairs = [
            # Arms
            (left_shoulder, left_elbow),
            (left_elbow, left_wrist),
            (right_shoulder, right_elbow),
            (right_elbow, right_wrist),
            # Legs
            (left_hip, left_knee),
            (left_knee, left_ankle),
            (right_hip, right_knee),
            (right_knee, right_ankle)
        ]

        for start, end in limb_pairs:
            cv2.line(canvas, start, end, color, thickness)

        # JOINTS - batch drawing
        joints = [
            left_shoulder, right_shoulder,
            left_elbow, right_elbow,
            left_wrist, right_wrist,
            left_hip, right_hip,
            left_knee, right_knee,
            left_ankle, right_ankle
        ]

        for joint in joints:
            cv2.circle(canvas, joint, joint_radius, color, -1)

        # DEBUG MARKERS
        if draw_debug_markers:
            draw_pose_debug_markers(canvas, landmarks, width, height)

    except Exception as e:
        print(f"Error drawing stickfigure: {e}")