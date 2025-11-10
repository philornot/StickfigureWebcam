import cv2
import mediapipe as mp
import numpy as np

"""Webcam stick figure application using MediaPipe Pose and Face Mesh.

Draws a simplified stick figure based on pose landmarks and indicates mouth
open state using Face Mesh landmarks.
"""


def calculate_mouth_openness(face_landmarks, width, height):
    """Determine whether the mouth is open based on Face Mesh landmarks.

    Uses key landmarks:
        13: upper lip (center)
        14: lower lip (center)

    Args:
        face_landmarks (Sequence | None): List of face landmarks (FaceMeshResult.multi_face_landmarks[0].landmark) or None.
        width (int): Frame width in pixels (reserved for future use).
        height (int): Frame height in pixels.

    Returns:
        bool: True if mouth detected as open, False otherwise.
    """
    if not face_landmarks:
        return False

    # Lip landmarks
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]

    # Convert relative coordinates to pixel space
    upper_y = upper_lip.y * height
    lower_y = lower_lip.y * height

    mouth_distance = abs(lower_y - upper_y)

    # Threshold (tunable). If vertical distance > 15 px -> mouth open.
    threshold = 15

    return mouth_distance > threshold


def draw_stickfigure(canvas, landmarks, width, height, mouth_open=False):
    """Draw a stick figure with optional mouth animation.

    Args:
        canvas (np.ndarray): Target BGR image to draw on.
        landmarks (Sequence): Pose landmarks (pose_landmarks.landmark).
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.
        mouth_open (bool): Whether to render the mouth as open.

    Raises:
        Exception: If landmark access fails.
    """
    if not landmarks:
        return

    def get_point(idx):
        lm = landmarks[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        return (x, y)

    color = (255, 255, 255)
    thickness = 3

    try:
        # HEAD (circle around nose landmark)
        nose = get_point(0)
        head_radius = 20
        cv2.circle(canvas, nose, head_radius, color, thickness)

        # MOUTH (centered below nose)
        mouth_y_offset = 10
        mouth_center = (nose[0], nose[1] + mouth_y_offset)

        if mouth_open:
            # Open mouth - filled circle
            cv2.circle(canvas, mouth_center, 5, color, -1)
        else:
            # Closed mouth - line
            cv2.line(
                canvas,
                (mouth_center[0] - 6, mouth_center[1]),
                (mouth_center[0] + 6, mouth_center[1]),
                color,
                2,
            )

        # EYES
        left_eye = (nose[0] - 8, nose[1] - 5)
        right_eye = (nose[0] + 8, nose[1] - 5)
        cv2.circle(canvas, left_eye, 2, color, -1)
        cv2.circle(canvas, right_eye, 2, color, -1)

        # TORSO
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_hip = get_point(23)
        right_hip = get_point(24)

        cv2.line(canvas, left_shoulder, right_shoulder, color, thickness)

        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2,
        )
        hip_center = (
            (left_hip[0] + right_hip[0]) // 2,
            (left_hip[1] + right_hip[1]) // 2,
        )
        cv2.line(canvas, shoulder_center, hip_center, color, thickness)
        cv2.line(canvas, left_hip, right_hip, color, thickness)

        # ARMS
        left_elbow = get_point(13)
        left_wrist = get_point(15)
        cv2.line(canvas, left_shoulder, left_elbow, color, thickness)
        cv2.line(canvas, left_elbow, left_wrist, color, thickness)

        right_elbow = get_point(14)
        right_wrist = get_point(16)
        cv2.line(canvas, right_shoulder, right_elbow, color, thickness)
        cv2.line(canvas, right_elbow, right_wrist, color, thickness)

        # LEGS
        left_knee = get_point(25)
        left_ankle = get_point(27)
        cv2.line(canvas, left_hip, left_knee, color, thickness)
        cv2.line(canvas, left_knee, left_ankle, color, thickness)

        right_knee = get_point(26)
        right_ankle = get_point(28)
        cv2.line(canvas, right_hip, right_knee, color, thickness)
        cv2.line(canvas, right_knee, right_ankle, color, thickness)

        # JOINT MARKERS
        joint_radius = 5
        joints = [
            left_shoulder, right_shoulder, left_elbow, right_elbow,
            left_wrist, right_wrist, left_hip, right_hip,
            left_knee, right_knee, left_ankle, right_ankle
        ]

        for joint in joints:
            cv2.circle(canvas, joint, joint_radius, color, -1)

    except Exception as e:
        print(f"⚠️ Drawing error: {e}")


def main():
    """Run the webcam stick figure application.

    Initializes MediaPipe Pose and Face Mesh, reads frames from default camera,
    computes mouth openness, renders a stick figure, overlays FPS and status,
    and exits on 'q'.
    """
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("✅ Camera started!")
    print(f"📐 Resolution: {width}x{height}")
    print("🎭 Move and open your mouth!")
    print("⌨️ Press 'q' to quit")

    # FPS counter setup
    fps_counter = 0
    fps_start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process both models
        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        # Mouth openness calculation
        mouth_open = calculate_mouth_openness(
            face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None,
            width,
            height
        )

        # Create drawing canvas
        stickfigure_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw stick figure or prompt
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
                'Stand in front of the camera!',
                (width // 2 - 200, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # FPS calculation (update every 30 frames)
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            fps = 30 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
            fps_start_time = fps_end_time
            fps_counter = 0

            cv2.putText(
                stickfigure_canvas,
                f'FPS: {fps:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Mouth status overlay
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

        cv2.imshow('Stickfigure Webcam', stickfigure_canvas)
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose.close()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Closing application")


if __name__ == "__main__":
    main()