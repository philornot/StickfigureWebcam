"""Simple webcam viewer using OpenCV.

This module opens the default webcam, displays live frames in a window,
and exits when the user presses 'q'.
"""

import cv2


def main():
    """Open the default camera and display frames until the user quits.

    The function attempts to open the default system camera (index 0). If the
    camera cannot be opened or frames cannot be read, an error message is
    printed and the function returns. While running, frames are shown in a
    window named "Stickfigure Webcam - Original". Pressing the 'q' key closes
    the window and releases the camera.

    Returns:
        None
    """
    # Open the default camera (0 = first camera in the system)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera!")
        print("Check that:")
        print("- Camera is connected")
        print("- You have permissions to the camera")
        print("- No other application is using the camera")
        return

    print("✅ Camera started! Press 'q' to quit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("❌ Cannot read frame from camera")
            break

        # Show the frame in a window
        cv2.imshow('Stickfigure Webcam - Camera', frame)

        # Wait 1ms for a key press, exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Closing application")


if __name__ == "__main__":
    main()
