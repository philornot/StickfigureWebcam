"""
Configuration settings for the Stickfigure Webcam application.

This module contains all configuration parameters used throughout the application,
including camera settings, detection parameters, and visual styling options.
"""

# -- Camera Settings --
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30  # Target FPS for virtual camera
PROCESSING_WIDTH = 320  # Smaller resolution for faster processing
PROCESSING_HEIGHT = 240

# -- MediaPipe Detection Settings --
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_MODEL_COMPLEXITY = 0  # 0=lite, 1=full, 2=heavy

FACE_MESH_MAX_FACES = 1
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# -- Mouth Detection Settings --
MOUTH_OPEN_THRESHOLD_RATIO = 0.025  # Percentage of face height
MOUTH_LANDMARKS = {
    'upper_lip_top': 13,
    'lower_lip_bottom': 14,
    'upper_outer_1': 78,
    'upper_outer_2': 308,
    'lower_outer_1': 87,
    'lower_outer_2': 317,
    'forehead': 10,
    'chin': 152
}

# -- Eye blink detection settings --
# Indices for left eye (vertical)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
# Indices for right eye (vertical)
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Ratio of (vertical eye dist / face height).
# A value below this threshold is considered "closed eyes".
EYES_CLOSED_RATIO_THRESHOLD = 0.05

# Number of consecutive frames eye must be closed to register.
EYES_CLOSED_CONSECUTIVE_FRAMES = 3

# -- Stickfigure drawing settings --
STICKFIGURE_COLOR = (255, 255, 255)  # White
STICKFIGURE_THICKNESS = 4
JOINT_RADIUS = 6
EYE_RADIUS_RATIO = 0.12  # Relative to head radius
MOUTH_WIDTH_RATIO = 0.5  # Relative to head radius
MOUTH_HEIGHT_RATIO = 0.25  # Relative to head radius

# -- Head proportions --
HEAD_RADIUS_RATIO = 0.4  # Relative to shoulder width
HEAD_RADIUS_MIN = 25
HEAD_RADIUS_MAX = 45
NECK_LENGTH_RATIO = 0.6  # Relative to head radius
EYE_Y_OFFSET_RATIO = 0.25  # Relative to head radius; positive moves eyes upwards in current formula
EYE_SPACING_RATIO = 0.35  # Relative to head radius
MOUTH_Y_OFFSET_RATIO = 0.4  # Relative to head radius
SHOULDER_CURVE_DEPTH_RATIO = 0.15  # Relative to shoulder width

# -- Debug Settings --
DEBUG_MARKER_COLORS = {
    'nose': (255, 0, 255),  # Magenta
    'arms': (255, 255, 0),  # Cyan
    'legs': (0, 255, 255),  # Yellow
    'other': (128, 128, 128)  # Gray
}

DEBUG_TEXT_COLOR = (255, 255, 255)
DEBUG_TEXT_SCALE = 0.7
DEBUG_TEXT_THICKNESS = 2

# -- FPS Calculation --
FPS_UPDATE_INTERVAL = 30  # Update FPS every N frames

# -- Window Names --
WINDOW_NAME_STICKFIGURE = 'Stickfigure Webcam'
WINDOW_NAME_DEBUG = 'Camera Debug'

# -- UI Messages --
MESSAGE_NO_PERSON = 'Human target not found'
MESSAGE_DEBUG_TOGGLE = 'Debug mode (press D to toggle)'
MESSAGE_QUIT = 'Press \'q\' to quit'
MESSAGE_DEBUG_KEY = 'Press \'d\' to toggle debug mode'
MESSAGE_MIRROR_KEY = "Press 'm' to toggle mirroring for virtual camera (helps with apps that mirror preview)"

# -- Virtual Camera Output Settings --
# When True, the virtual camera image is horizontally mirrored.
# Set to False if your app does not mirror the local preview, so text/logos stay readable.
# Note: Apps handle local preview vs. transmitted video differently; choose what remote viewers should see.
VCAM_MIRROR_OUTPUT = True
