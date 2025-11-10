"""
Camera selection GUI module.

Provides a simple GUI for users to select their camera before starting
the application. Supports both tkinter and PyQt5 backends.
"""

import cv2


def detect_available_cameras(max_cameras=10):
    """Detect all available camera devices.

    Attempts to open cameras from index 0 to max_cameras and returns
    a list of working camera indices with their names.

    Args:
        max_cameras: Maximum number of camera indices to check.

    Returns:
        list: List of tuples (index, name) for available cameras.
    """
    available_cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get camera name (platform-dependent)
            ret, _ = cap.read()
            if ret:
                # Get camera backend info
                backend = cap.getBackendName()
                name = f"Camera {i} ({backend})"
                available_cameras.append((i, name))
            cap.release()

    return available_cameras


def select_camera_tkinter(available_cameras):
    """Show camera selection dialog using tkinter.

    Args:
        available_cameras: List of tuples (index, name) for available cameras.

    Returns:
        int or None: Selected camera index, or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        print("Error: tkinter not available")
        return None

    selected_camera = [None]  # Use list to allow modification in nested function

    def on_select():
        """Handle camera selection."""
        selection = camera_listbox.curselection()
        if selection:
            idx = selection[0]
            selected_camera[0] = available_cameras[idx][0]
            root.quit()
        else:
            messagebox.showwarning("No Selection", "Please select a camera")

    def on_cancel():
        """Handle cancel button."""
        root.quit()

    # Create main window
    root = tk.Tk()
    root.title("Select Camera - Stickfigure Webcam")
    root.geometry("400x300")
    root.resizable(False, False)

    # Title label
    title_label = tk.Label(
        root,
        text="Select Camera",
        font=("Arial", 14, "bold")
    )
    title_label.pack(pady=10)

    # Instructions
    instruction_label = tk.Label(
        root,
        text="Choose a camera device to use:",
        font=("Arial", 10)
    )
    instruction_label.pack(pady=5)

    # Camera list frame with scrollbar
    list_frame = tk.Frame(root)
    list_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    camera_listbox = tk.Listbox(
        list_frame,
        yscrollcommand=scrollbar.set,
        font=("Arial", 10),
        selectmode=tk.SINGLE,
        height=8
    )
    camera_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=camera_listbox.yview)

    # Populate list
    for idx, name in available_cameras:
        camera_listbox.insert(tk.END, name)

    # Select first camera by default
    if available_cameras:
        camera_listbox.select_set(0)

    # Double-click to select
    camera_listbox.bind('<Double-Button-1>', lambda e: on_select())

    # Button frame
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    # Start button
    start_button = tk.Button(
        button_frame,
        text="Start",
        command=on_select,
        width=10,
        font=("Arial", 10),
        bg="#4CAF50",
        fg="white"
    )
    start_button.pack(side=tk.LEFT, padx=5)

    # Cancel button
    cancel_button = tk.Button(
        button_frame,
        text="Cancel",
        command=on_cancel,
        width=10,
        font=("Arial", 10)
    )
    cancel_button.pack(side=tk.LEFT, padx=5)

    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    # Run dialog
    root.mainloop()
    root.destroy()

    return selected_camera[0]


def select_camera_console(available_cameras):
    """Show camera selection in console (fallback).

    Args:
        available_cameras: List of tuples (index, name) for available cameras.

    Returns:
        int or None: Selected camera index, or None if cancelled.
    """
    print("\n" + "=" * 50)
    print("STICKFIGURE WEBCAM - CAMERA SELECTION")
    print("=" * 50)
    print("\nAvailable cameras:")

    for i, (idx, name) in enumerate(available_cameras):
        print(f"  [{i + 1}] {name}")

    print(f"  [0] Cancel")
    print()

    while True:
        try:
            choice = input("Select camera number (or 0 to cancel): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                return None

            if 1 <= choice_num <= len(available_cameras):
                selected_idx = available_cameras[choice_num - 1][0]
                print(f"\nSelected: {available_cameras[choice_num - 1][1]}")
                return selected_idx
            else:
                print(f"Please enter a number between 0 and {len(available_cameras)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            return None


def show_camera_selector(use_gui=True):
    """Show camera selection dialog.

    Args:
        use_gui: If True, attempt to show GUI dialog. Falls back to console if unavailable.

    Returns:
        int or None: Selected camera index, or None if cancelled/no cameras found.
    """
    print("Detecting available cameras...")
    available_cameras = detect_available_cameras()

    if not available_cameras:
        print("\nError: No cameras detected!")
        print("Please check that:")
        print("  - Your camera is connected")
        print("  - Camera drivers are installed")
        print("  - Camera is not being used by another application")
        return None

    print(f"Found {len(available_cameras)} camera(s)")

    # If only one camera, auto-select it
    if len(available_cameras) == 1:
        print(f"Auto-selecting: {available_cameras[0][1]}")
        return available_cameras[0][0]

    # Multiple cameras - show selection dialog
    if use_gui:
        try:
            return select_camera_tkinter(available_cameras)
        except Exception as e:
            print(f"GUI unavailable ({e}), falling back to console...")
            return select_camera_console(available_cameras)
    else:
        return select_camera_console(available_cameras)


def test_camera_selector():
    """Test the camera selector."""
    camera_id = show_camera_selector(use_gui=True)

    if camera_id is None:
        print("No camera selected")
        return

    print(f"\nTesting camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Could not open selected camera")
        return

    print("Camera opened successfully!")
    print("Press 'q' to quit test")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f"Camera {camera_id} Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete")


if __name__ == "__main__":
    test_camera_selector()
