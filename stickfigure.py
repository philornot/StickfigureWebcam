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
    """
    Draw curved shoulder line using quadratic bezier curve.

    Args:
        canvas: Canvas to draw on
        left_shoulder: (x, y) position of left shoulder
        right_shoulder: (x, y) position of right shoulder
        shoulder_center: (x, y) center position between shoulders
        shoulder_width: Distance between shoulders
    """
    color = config.STICKFIGURE_COLOR
    thickness = config.STICKFIGURE_THICKNESS

    # Calculate control point for the curve
    shoulder_curve_depth = int(shoulder_width * config.SHOULDER_CURVE_DEPTH_RATIO)
    control_point = (shoulder_center[0], shoulder_center[1] + shoulder_curve_depth)

    # Create curved shoulder using quadratic bezier curve
    num_points = 20
    shoulder_curve_points = []
    for i in range(num_points + 1):
        t = i / num_points
        # Quadratic Bezier formula: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        x = int((1 - t) ** 2 * left_shoulder[0] +
                2 * (1 - t) * t * control_point[0] +
                t ** 2 * right_shoulder[0])
        y = int((1 - t) ** 2 * left_shoulder[1] +
                2 * (1 - t) * t * control_point[1] +
                t ** 2 * right_shoulder[1])
        shoulder_curve_points.append([x, y])

    # Draw the curved shoulder line
    shoulder_curve_points = np.array(shoulder_curve_points, np.int32)
    cv2.polylines(canvas, [shoulder_curve_points], False, color, thickness)


def draw_pose_debug_markers(canvas, landmarks, width, height):
    """
    Draw debug markers showing all detected landmark positions.

    Args:
        canvas: Canvas to draw on
        landmarks: MediaPipe pose landmarks
        width: Canvas width
        height: Canvas height
    """
    debug_colors = config.DEBUG_MARKER_COLORS

    # Draw all 33 pose landmarks as small colored circles
    for idx in range(33):
        try:
            point = get_point(landmarks, idx, width, height)

            if idx == 0:  # Nose
                marker_color = debug_colors['nose']
            elif idx in [11, 12, 13, 14, 15, 16]:  # Arms
                marker_color = debug_colors['arms']
            elif idx in [23, 24, 25, 26, 27, 28]:  # Legs
                marker_color = debug_colors['legs']
            else:  # Other points
                marker_color = debug_colors['other']

            cv2.circle(canvas, point, 3, marker_color, -1)
            cv2.putText(canvas, str(idx),
                        (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        marker_color, 1)
        except:
            pass


def draw_stickfigure(canvas, landmarks, width, height, mouth_open=False,
                     eyes_closed=False, draw_debug_markers=False):
    """
    Draw a stick figure based on pose landmarks with natural proportions.

    Args:
        canvas: Numpy array to draw on
        landmarks: MediaPipe pose landmarks
        width: Canvas width
        height: Canvas height
        mouth_open: Whether mouth is open
        eyes_closed: Whether eyes are closed
        draw_debug_markers: Whether to draw small markers at detected landmark positions
    """
    if not landmarks:
        return

    color = config.STICKFIGURE_COLOR
    thickness = config.STICKFIGURE_THICKNESS
    joint_radius = config.JOINT_RADIUS

    try:
        # Key body points
        left_shoulder = get_point(landmarks, 11, width, height)
        right_shoulder = get_point(landmarks, 12, width, height)
        left_elbow = get_point(landmarks, 13, width, height)
        right_elbow = get_point(landmarks, 14, width, height)
        left_wrist = get_point(landmarks, 15, width, height)
        right_wrist = get_point(landmarks, 16, width, height)
        left_hip = get_point(landmarks, 23, width, height)
        right_hip = get_point(landmarks, 24, width, height)
        left_knee = get_point(landmarks, 25, width, height)
        right_knee = get_point(landmarks, 26, width, height)
        left_ankle = get_point(landmarks, 27, width, height)
        right_ankle = get_point(landmarks, 28, width, height)

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

        # ARMS
        cv2.line(canvas, left_shoulder, left_elbow, color, thickness)
        cv2.line(canvas, left_elbow, left_wrist, color, thickness)
        cv2.line(canvas, right_shoulder, right_elbow, color, thickness)
        cv2.line(canvas, right_elbow, right_wrist, color, thickness)

        # LEGS
        cv2.line(canvas, left_hip, left_knee, color, thickness)
        cv2.line(canvas, left_knee, left_ankle, color, thickness)
        cv2.line(canvas, right_hip, right_knee, color, thickness)
        cv2.line(canvas, right_knee, right_ankle, color, thickness)

        # JOINTS
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
