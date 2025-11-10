import cv2
import mediapipe as mp


def main():
    """
    Main function that initializes webcam pose detection using MediaPipe.

    This function captures video from the default camera (index 0), detects human pose
    landmarks using MediaPipe Pose, and displays the results in real-time with visual
    overlays. Press 'q' to quit the application.

    The pose detection uses:
        - min_detection_confidence: 0.5 (confidence threshold for person detection)
        - min_tracking_confidence: 0.5 (confidence threshold for tracking between frames)
        - model_complexity: 1 (balance between speed and accuracy)
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Create Pose object with configuration:
    # - min_detection_confidence: confidence that a person is detected (0-1)
    # - min_tracking_confidence: tracking confidence between frames
    # - model_complexity: 0=fastest, 2=most accurate (we use 1 for balance)
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera!")
        return

    print("✅ Camera started! Press 'q' to quit.")
    print("📊 Detecting body pose...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe requires RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame through pose detection model
        results = pose.process(frame_rgb)

        # Draw detected landmarks on original image (for debugging)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Display info about number of detected points
            num_landmarks = len(results.pose_landmarks.landmark)
            cv2.putText(frame, f'Landmarks: {num_landmarks}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Stickfigure Webcam - Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Closing application")


if __name__ == "__main__":
    main()
