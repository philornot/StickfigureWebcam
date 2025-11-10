"""Stick-figure pose visualizer using OpenCV and MediaPipe.

Opens the default webcam, tracks human pose, and renders a simple stick figure
on a separate black canvas next to the original camera feed. Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np


def draw_stickfigure(canvas, landmarks, width, height):
    """Draw a simple stick figure based on MediaPipe pose landmarks.

    The pose model provides 33 landmarks, but only the key joints are used:
    0: nose
    11, 12: shoulders (left, right)
    13, 14: elbows
    15, 16: wrists
    23, 24: hips
    25, 26: knees
    27, 28: ankles

    Args:
        canvas: Numpy image array (H, W, 3) in BGR where the figure is drawn.
        landmarks: Sequence of normalized landmarks with `.x` and `.y` in [0, 1].
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        None. Draws in-place on the provided canvas.
    """
    if not landmarks:
        return

    # Map normalized coordinates [0..1] to pixel coordinates
    def get_point(idx):
        lm = landmarks[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        return (x, y)

    # Color and line thickness
    color = (255, 255, 255)  # white
    thickness = 3

    try:
        # HEAD - draw a circle around the nose
        nose = get_point(0)
        cv2.circle(canvas, nose, 20, color, thickness)

        # TORSO
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_hip = get_point(23)
        right_hip = get_point(24)

        # Shoulder line
        cv2.line(canvas, left_shoulder, right_shoulder, color, thickness)

        # Spine: center of shoulders to center of hips
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2,
        )
        hip_center = (
            (left_hip[0] + right_hip[0]) // 2,
            (left_hip[1] + right_hip[1]) // 2,
        )
        cv2.line(canvas, shoulder_center, hip_center, color, thickness)

        # Hip line
        cv2.line(canvas, left_hip, right_hip, color, thickness)

        # LEFT ARM
        left_elbow = get_point(13)
        left_wrist = get_point(15)
        cv2.line(canvas, left_shoulder, left_elbow, color, thickness)
        cv2.line(canvas, left_elbow, left_wrist, color, thickness)

        # RIGHT ARM
        right_elbow = get_point(14)
        right_wrist = get_point(16)
        cv2.line(canvas, right_shoulder, right_elbow, color, thickness)
        cv2.line(canvas, right_elbow, right_wrist, color, thickness)

        # LEFT LEG
        left_knee = get_point(25)
        left_ankle = get_point(27)
        cv2.line(canvas, left_hip, left_knee, color, thickness)
        cv2.line(canvas, left_knee, left_ankle, color, thickness)

        # RIGHT LEG
        right_knee = get_point(26)
        right_ankle = get_point(28)
        cv2.line(canvas, right_hip, right_knee, color, thickness)
        cv2.line(canvas, right_knee, right_ankle, color, thickness)

        # Joint dots for a nicer effect
        joint_radius = 5
        joints = [
            left_shoulder, right_shoulder, left_elbow, right_elbow,
            left_wrist, right_wrist, left_hip, right_hip,
            left_knee, right_knee, left_ankle, right_ankle
        ]
        for joint in joints:
            cv2.circle(canvas, joint, joint_radius, color, -1)

    except Exception as e:
        # Some points may be out of frame or unavailable
        print(f"⚠️  Drawing error: {e}")


def main():
    """Run the webcam loop and render both the original camera feed and stick figure.

    Initializes MediaPipe Pose, reads frames from the default camera, mirrors
    the image for an intuitive experience, and draws a stick figure using
    detected landmarks on a separate black canvas.

    Press 'q' to quit.

    Returns:
        None
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera!")
        return

    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("✅ Camera started!")
    print(f"📐 Resolution: {width}x{height}")
    print("🎭 Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror flip (more intuitive)
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Create black canvas for the stick figure
        stickfigure_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw stick figure if a pose is detected
        if results.pose_landmarks:
            draw_stickfigure(
                stickfigure_canvas,
                results.pose_landmarks.landmark,
                width,
                height,
            )
        else:
            # If no person detected, show a message
            cv2.putText(
                stickfigure_canvas,
                "Stand in front of the camera!",
                (width // 2 - 200, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Show both windows for comparison
        cv2.imshow("Camera", frame)
        cv2.imshow("Stickfigure", stickfigure_canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Closing application")


if __name__ == "__main__":
    main()
