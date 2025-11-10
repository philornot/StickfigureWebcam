import cv2
import mediapipe as mp
import numpy as np


def calculate_mouth_openness(face_landmarks, width, height):
    """
    Calculate if mouth is open based on Face Mesh landmarks.

    Args:
        face_landmarks: MediaPipe face mesh landmarks
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        bool: True if mouth is open, False otherwise
    """
    if not face_landmarks:
        return False

    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]

    upper_y = upper_lip.y * height
    lower_y = lower_lip.y * height

    mouth_distance = abs(lower_y - upper_y)
    threshold = 15

    return mouth_distance > threshold


def draw_stickfigure(canvas, landmarks, width, height, mouth_open=False):
    """
    Draw a stick figure based on pose landmarks.

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
        """Convert normalized landmark to pixel coordinates."""
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

        # HEAD - position it just above shoulders with short neck
        head_radius = 36
        # Head center is slightly above shoulder center
        head_center = (shoulder_center[0], shoulder_center[1] - head_radius - 8)

        cv2.circle(canvas, head_center, head_radius, color, thickness)

        # VERY SHORT NECK - just connect bottom of head to shoulders
        neck_bottom = (head_center[0], head_center[1] + head_radius)
        cv2.line(canvas, neck_bottom, shoulder_center, color, thickness)

        # FACE - eyes and mouth
        left_eye = (head_center[0] - 7, head_center[1] - 3)
        right_eye = (head_center[0] + 7, head_center[1] - 3)
        cv2.circle(canvas, left_eye, 3, color, -1)
        cv2.circle(canvas, right_eye, 3, color, -1)

        mouth_center = (head_center[0], head_center[1] + 7)
        if mouth_open:
            cv2.circle(canvas, mouth_center, 5, color, -1)
        else:
            cv2.line(canvas,
                     (mouth_center[0] - 6, mouth_center[1]),
                     (mouth_center[0] + 6, mouth_center[1]),
                     color, 2)

        # TORSO
        cv2.line(canvas, left_shoulder, right_shoulder, color, thickness)
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
    """Main application loop."""
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

    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        processing_frame = cv2.resize(frame, (320, 240))
        frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        mouth_open = False
        if face_results.multi_face_landmarks:
            mouth_open = calculate_mouth_openness(
                face_results.multi_face_landmarks[0].landmark,
                320,
                240
            )

        stickfigure_canvas = np.zeros((height, width, 3), dtype=np.uint8)

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

        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            current_fps = 30 / time_diff
            fps_start_time = fps_end_time
            fps_counter = 0

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

        cv2.imshow('Stickfigure Webcam', stickfigure_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose.close()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
