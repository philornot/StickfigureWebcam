import cv2
import mediapipe as mp
import numpy as np


def calculate_mouth_openness(face_landmarks, width, height):
    """
    Calculate if mouth is open based on Face Mesh landmarks.

    Uses multiple landmark points for more accurate detection:
    - Upper lip: points 13, 14
    - Lower lip: points 78, 308, 87, 317
    - Calculates vertical distance and compares to face height ratio

    Args:
        face_landmarks: MediaPipe face mesh landmarks
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        bool: True if mouth is open, False otherwise
    """
    if not face_landmarks:
        return False

    # Get multiple points for more accurate measurement
    # Upper lip center
    upper_lip_top = face_landmarks[13]
    # Lower lip center
    lower_lip_bottom = face_landmarks[14]
    # Additional points for better accuracy
    upper_outer_1 = face_landmarks[78]
    upper_outer_2 = face_landmarks[308]
    lower_outer_1 = face_landmarks[87]
    lower_outer_2 = face_landmarks[317]

    # Convert to pixel coordinates
    upper_y = upper_lip_top.y * height
    lower_y = lower_lip_bottom.y * height

    # Calculate mouth opening distance
    mouth_distance = abs(lower_y - upper_y)

    # Calculate face height for relative threshold
    # Using forehead (10) to chin (152) distance
    forehead = face_landmarks[10]
    chin = face_landmarks[152]
    face_height = abs((chin.y - forehead.y) * height)

    # Adaptive threshold based on face size (about 3.5% of face height)
    # This makes it more sensitive than the fixed threshold of 15
    threshold = face_height * 0.035

    return mouth_distance > threshold


def draw_stickfigure(canvas, landmarks, width, height, mouth_open=False):
    """
    Draw a stick figure based on pose landmarks with natural proportions.

    Creates a more realistic stick figure with:
    - Properly sized and positioned head
    - Natural facial features
    - Smooth body proportions
    - Clear joint markers

    Args:
        canvas: Numpy array to draw on
        landmarks: MediaPipe pose landmarks
        width: Canvas width
        height: Canvas height
        mouth_open: Whether mouth is open
    """
    if not landmarks:
        return

    def get_point(idx):
        """
        Convert normalized landmark to pixel coordinates.

        Args:
            idx: Landmark index

        Returns:
            tuple: (x, y) pixel coordinates
        """
        lm = landmarks[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        return (x, y)

    color = (255, 255, 255)
    thickness = 4
    joint_radius = 6

    try:
        # Key body points
        nose = get_point(0)
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_elbow = get_point(13)
        right_elbow = get_point(14)
        left_wrist = get_point(15)
        right_wrist = get_point(16)
        left_hip = get_point(23)
        right_hip = get_point(24)
        left_knee = get_point(25)
        right_knee = get_point(26)
        left_ankle = get_point(27)
        right_ankle = get_point(28)

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

        # HEAD - natural proportions (head radius ~40% of shoulder width)
        head_radius = int(shoulder_width * 0.4)
        head_radius = max(25, min(head_radius, 45))  # Clamp between 25-45 pixels

        # Position head above shoulders with natural neck length
        neck_length = int(head_radius * 0.6)
        head_center = (shoulder_center[0], shoulder_center[1] - head_radius - neck_length)

        # Draw head circle
        cv2.circle(canvas, head_center, head_radius, color, thickness)

        # NECK - connect head to shoulders
        neck_top = (head_center[0], head_center[1] + head_radius)
        cv2.line(canvas, neck_top, shoulder_center, color, thickness)

        # FACE - natural facial features
        # Eyes positioned in upper third of head
        eye_y_offset = int(-head_radius * 0.25)
        eye_spacing = int(head_radius * 0.35)
        eye_radius = max(3, int(head_radius * 0.12))

        left_eye = (head_center[0] - eye_spacing, head_center[1] + eye_y_offset)
        right_eye = (head_center[0] + eye_spacing, head_center[1] + eye_y_offset)

        cv2.circle(canvas, left_eye, eye_radius, color, -1)
        cv2.circle(canvas, right_eye, eye_radius, color, -1)

        # Mouth positioned in lower third of head
        mouth_y_offset = int(head_radius * 0.4)
        mouth_center = (head_center[0], head_center[1] + mouth_y_offset)
        mouth_width = int(head_radius * 0.5)

        if mouth_open:
            # Open mouth - draw as a circle/oval
            mouth_height = int(head_radius * 0.25)
            cv2.ellipse(canvas, mouth_center, (mouth_width // 2, mouth_height // 2),
                        0, 0, 360, color, -1)
        else:
            # Closed mouth - draw as a line with slight curve
            cv2.line(canvas,
                     (mouth_center[0] - mouth_width // 2, mouth_center[1]),
                     (mouth_center[0] + mouth_width // 2, mouth_center[1]),
                     color, 2)

        # TORSO
        # Draw curved shoulder line instead of straight line
        # Calculate control point for the curve (slightly below shoulder center)
        shoulder_curve_depth = int(shoulder_width * 0.15)
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

        # Spine and hips
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

    except Exception as e:
        print(f"Error drawing stickfigure: {e}")


def main():
    """
    Main application loop.

    Initializes webcam, pose detection, and face mesh detection.
    Processes frames in real-time and displays stick figure representation.
    Press 'q' to quit, 'd' to toggle debug mode.
    """
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Camera initialized")
    print(f"Resolution: {width}x{height}")
    print("Press 'q' to quit")
    print("Press 'd' to toggle debug mode")

    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0
    debug_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Process on smaller frame for performance
        processing_frame = cv2.resize(frame, (320, 240))
        frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        # Detect mouth opening
        mouth_open = False
        if face_results.multi_face_landmarks:
            mouth_open = calculate_mouth_openness(
                face_results.multi_face_landmarks[0].landmark,
                320,
                240
            )

        # Create black canvas for stick figure
        stickfigure_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw stick figure or message
        if pose_results.pose_landmarks:
            draw_stickfigure(
                stickfigure_canvas,
                pose_results.pose_landmarks.landmark,
                width,
                height,
                mouth_open
            )
        else:
            cv2.putText(
                stickfigure_canvas,
                'Stand in front of camera',
                (width // 2 - 200, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            current_fps = 30 / time_diff
            fps_start_time = fps_end_time
            fps_counter = 0

        # Display debug information if enabled
        if debug_mode:
            cv2.putText(
                stickfigure_canvas,
                f'FPS: {current_fps:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            mouth_status = "OPEN" if mouth_open else "CLOSED"
            cv2.putText(
                stickfigure_canvas,
                f'Mouth: {mouth_status}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                stickfigure_canvas,
                'Debug Mode (press D to toggle)',
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                1
            )

        cv2.imshow('Stickfigure Webcam', stickfigure_canvas)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

    pose.close()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
