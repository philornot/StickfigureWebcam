# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import Any, Dict, List, Tuple

import cv2

# Try to import pyvirtualcam, but don't react to error
try:
    import pyvirtualcam

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False

# Try to import mediapipe
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class SystemCheck:
    """
    Class for checking availability and correctness of configuration
    of essential application components.
    """

    def __init__(self, logger=None):
        """
        Initializes system check.

        Args:
            logger: Optional logger for recording check results
        """
        self.logger = logger
        self.system = platform.system()  # 'Windows', 'Linux', 'Darwin' (macOS)

        # Check results
        self.results = {
            "camera": {"status": False, "message": "", "details": {}},
            "virtual_camera": {"status": False, "message": "", "details": {}},
            "mediapipe": {"status": False, "message": "", "details": {}},
            "obs": {"status": False, "message": "", "details": {}},
            "v4l2loopback": {"status": False, "message": "", "details": {}},
        }

        # Component installation links
        self.install_links = {
            "obs": "https://obsproject.com/download",
            "v4l2loopback": "https://github.com/umlaeute/v4l2loopback",
            "pyvirtualcam": "https://pypi.org/project/pyvirtualcam/",
            "mediapipe": "https://pypi.org/project/mediapipe/",
            "obs_virtualcam_plugin_mac": "https://github.com/johnboiles/obs-mac-virtualcam",
        }

    def check_all(self) -> Dict[str, Any]:
        """
        Performs all system checks.

        Returns:
            Dict[str, Any]: Dictionary with check results
        """
        self._log("Starting system check...")

        # Camera check
        self.check_camera()

        # Virtual camera check
        self.check_virtual_camera()

        # MediaPipe check
        self.check_mediapipe()

        # OBS check (only on Windows and macOS)
        if self.system in ["Windows", "Darwin"]:
            self.check_obs()

        # v4l2loopback check (only on Linux)
        if self.system == "Linux":
            self.check_v4l2loopback()

        self._log("System check completed")

        return self.results

    def check_camera(self, camera_id: int = 0) -> Dict[str, Any]:
        """
        Checks if camera is available and working correctly.

        Args:
            camera_id (int): Camera identifier to check

        Returns:
            Dict[str, Any]: Check result
        """
        self._log(f"Checking camera (ID: {camera_id})...")

        result = self.results["camera"]
        result["details"]["camera_id"] = camera_id

        try:
            # Attempt to open camera
            cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                result["status"] = False
                result["message"] = f"Cannot open camera with ID: {camera_id}"
                self._log(result["message"], level="WARNING")
            else:
                # Check if frame can be read
                ret, frame = cap.read()

                if not ret:
                    result["status"] = False
                    result["message"] = (
                        f"Camera with ID: {camera_id} is available but cannot read frame"
                    )
                    self._log(result["message"], level="WARNING")
                else:
                    # Read camera parameters
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    result["status"] = True
                    result["message"] = f"Camera with ID: {camera_id} is working correctly"
                    result["details"].update({"width": width, "height": height, "fps": fps})
                    self._log(result["message"])

                # Close camera
                cap.release()
        except Exception as e:
            result["status"] = False
            result["message"] = f"Error during camera check: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_virtual_camera(self) -> Dict[str, Any]:
        """
        Checks if virtual camera is available and configured.

        Returns:
            Dict[str, Any]: Check result
        """
        self._log("Checking virtual camera...")

        result = self.results["virtual_camera"]

        # First check if pyvirtualcam is installed
        if not PYVIRTUALCAM_AVAILABLE:
            result["status"] = False
            result["message"] = "pyvirtualcam library is not installed"
            result["details"]["install_command"] = "pip install pyvirtualcam"
            result["details"]["install_link"] = self.install_links["pyvirtualcam"]
            self._log(result["message"], level="WARNING")
            return result

        try:
            # Check available backends - function may not be available in older versions
            available_backends = []
            try:
                # Attempt to use get_available_backends (newer pyvirtualcam versions)
                if hasattr(pyvirtualcam, "get_available_backends"):
                    available_backends = pyvirtualcam.get_available_backends()
                    result["details"]["available_backends"] = available_backends
            except Exception:
                pass

            # Attempt to create virtual camera
            try:
                # Use small resolution for test
                cam = pyvirtualcam.Camera(width=320, height=240, fps=20)
                cam_info = {
                    "backend": cam.backend,
                    "width": cam.width,
                    "height": cam.height,
                    "fps": cam.fps,
                    "device": getattr(cam, "device", None),
                }
                cam.close()

                result["status"] = True
                result["message"] = (
                    f"Virtual camera is working correctly (backend: {cam_info['backend']})"
                )
                result["details"].update(cam_info)
                self._log(result["message"])

            except Exception as e:
                result["status"] = False
                result["message"] = f"Error during virtual camera check: {str(e)}"
                result["details"]["error"] = str(e)
                self._log(result["message"], level="WARNING")

                # Add suggestions depending on system
                if self.system == "Windows":
                    if "OBS" in str(e):
                        result["details"][
                            "suggestion"
                        ] = "Install OBS Studio and start Virtual Camera"
                        result["details"]["install_link"] = self.install_links["obs"]
                elif self.system == "Linux":
                    if "v4l2loopback" in str(e):
                        result["details"]["suggestion"] = "Install and load v4l2loopback module"
                        result["details"]["install_link"] = self.install_links["v4l2loopback"]
                elif self.system == "Darwin":  # macOS
                    result["details"][
                        "suggestion"
                    ] = "Install OBS Studio and obs-mac-virtualcam plugin"
                    result["details"]["install_link_obs"] = self.install_links["obs"]
                    result["details"]["install_link_plugin"] = self.install_links[
                        "obs_virtualcam_plugin_mac"
                    ]

        except Exception as e:
            result["status"] = False
            result["message"] = f"Error during virtual camera check: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_mediapipe(self) -> Dict[str, Any]:
        """
        Checks if MediaPipe is available and working correctly.

        Returns:
            Dict[str, Any]: Check result
        """
        self._log("Checking MediaPipe...")

        result = self.results["mediapipe"]

        if not MEDIAPIPE_AVAILABLE:
            result["status"] = False
            result["message"] = "MediaPipe library is not installed"
            result["details"]["install_command"] = "pip install mediapipe"
            result["details"]["install_link"] = self.install_links["mediapipe"]
            self._log(result["message"], level="WARNING")
            return result

        try:
            # Attempt to create pose detector
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,  # Use simplest model for test
                min_detection_confidence=0.5,
            )

            # Check MediaPipe version
            mp_version = mp.__version__

            result["status"] = True
            result["message"] = f"MediaPipe is working correctly (version: {mp_version})"
            result["details"]["version"] = mp_version
            self._log(result["message"])

            # Close detector
            pose.close()

        except Exception as e:
            result["status"] = False
            result["message"] = f"Error during MediaPipe check: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_obs(self) -> Dict[str, Any]:
        """
        Checks if OBS Studio is installed (only Windows/macOS).

        Returns:
            Dict[str, Any]: Check result
        """
        self._log("Checking OBS Studio...")

        result = self.results["obs"]

        # Check only on Windows and macOS
        if self.system not in ["Windows", "Darwin"]:
            result["status"] = None
            result["message"] = "OBS check skipped - unsupported system"
            return result

        try:
            # OBS installation paths
            obs_paths = []

            if self.system == "Windows":
                # Typical Windows locations
                program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
                program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")

                obs_paths = [
                    os.path.join(program_files, "obs-studio"),
                    os.path.join(program_files_x86, "obs-studio"),
                ]

                # For Steam
                steam_path = os.path.join(
                    program_files, "Steam", "steamapps", "common", "obs-studio"
                )
                if os.path.exists(steam_path):
                    obs_paths.append(steam_path)

            elif self.system == "Darwin":  # macOS
                # Typical macOS locations
                obs_paths = ["/Applications/OBS.app", os.path.expanduser("~/Applications/OBS.app")]

            # Check if OBS exists in any location
            obs_installed = False
            obs_location = None

            for path in obs_paths:
                if os.path.exists(path):
                    obs_installed = True
                    obs_location = path
                    break

            if obs_installed:
                result["status"] = True
                result["message"] = "OBS Studio is installed"
                result["details"]["location"] = obs_location
                self._log(result["message"])
            else:
                result["status"] = False
                result["message"] = "OBS Studio is not installed"
                result["details"]["install_link"] = self.install_links["obs"]
                self._log(result["message"], level="WARNING")

        except Exception as e:
            result["status"] = None
            result["message"] = f"Error during OBS check: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_v4l2loopback(self) -> Dict[str, Any]:
        """
        Checks if v4l2loopback module is installed and loaded (only Linux).

        Returns:
            Dict[str, Any]: Check result
        """
        self._log("Checking v4l2loopback module...")

        result = self.results["v4l2loopback"]

        # Check only on Linux
        if self.system != "Linux":
            result["status"] = None
            result["message"] = "v4l2loopback check skipped - unsupported system"
            return result

        try:
            # Check if module is loaded
            module_loaded = False
            loaded_modules = subprocess.check_output(["lsmod"]).decode("utf-8")

            if "v4l2loopback" in loaded_modules:
                module_loaded = True

            # Check if virtual camera devices exist
            v4l_devices = []

            try:
                devices_output = subprocess.check_output(["ls", "-la", "/dev/video*"]).decode(
                    "utf-8"
                )
                v4l_devices = [line for line in devices_output.splitlines() if "video" in line]
            except subprocess.CalledProcessError:
                pass

            if module_loaded:
                result["status"] = True
                result["message"] = "v4l2loopback module is loaded"
                result["details"]["devices"] = len(v4l_devices)
                self._log(result["message"])
            else:
                result["status"] = False
                result["message"] = "v4l2loopback module is not loaded"
                result["details"]["install_command"] = "sudo apt-get install v4l2loopback-dkms"
                result["details"]["load_command"] = "sudo modprobe v4l2loopback"
                result["details"]["install_link"] = self.install_links["v4l2loopback"]
                self._log(result["message"], level="WARNING")

        except Exception as e:
            result["status"] = None
            result["message"] = f"Error during v4l2loopback check: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def get_missing_components(self) -> List[Dict[str, Any]]:
        """
        Returns list of missing or incorrectly configured components.

        Returns:
            List[Dict[str, Any]]: List of missing components with information
        """
        missing = []

        for component, result in self.results.items():
            if result["status"] is False:  # Skip None (unsupported) and True (OK)
                missing.append(
                    {"name": component, "message": result["message"], "details": result["details"]}
                )

        return missing

    def are_all_requirements_met(self) -> bool:
        """
        Checks if all required components are available.

        Returns:
            bool: True if all requirements are met
        """
        # Check only components required for given system
        required_components = ["camera", "virtual_camera", "mediapipe"]

        if self.system == "Windows":
            required_components.append("obs")
        elif self.system == "Linux":
            required_components.append("v4l2loopback")

        for component in required_components:
            if component in self.results and self.results[component]["status"] is False:
                return False

        return True

    def get_installation_instructions(self) -> Dict[str, List[str]]:
        """
        Generates installation instructions for missing components.

        Returns:
            Dict[str, List[str]]: Dictionary of instructions for different components
        """
        instructions = {}

        # Camera instructions
        if not self.results["camera"]["status"]:
            instructions["camera"] = [
                "Check if camera is connected and working correctly.",
                "Make sure no other application is using the camera.",
                "Check system privacy settings and application permissions to use camera.",
            ]

        # Virtual camera instructions
        if not self.results["virtual_camera"]["status"]:
            if self.system == "Windows":
                instructions["virtual_camera"] = [
                    f"Install OBS Studio: {self.install_links['obs']}",
                    "Run OBS Studio and enable Virtual Camera (Tools -> Start Virtual Camera)",
                    "If you already have OBS, make sure Virtual Camera is enabled",
                ]
            elif self.system == "Linux":
                instructions["virtual_camera"] = [
                    f"Install v4l2loopback: {self.install_links['v4l2loopback']}",
                    "Install via apt: sudo apt-get install v4l2loopback-dkms",
                    "Load module: sudo modprobe v4l2loopback",
                ]
            elif self.system == "Darwin":  # macOS
                instructions["virtual_camera"] = [
                    f"Install OBS Studio: {self.install_links['obs']}",
                    f"Install obs-mac-virtualcam plugin: {self.install_links['obs_virtualcam_plugin_mac']}",
                    "Run OBS Studio and enable Virtual Camera (Tools -> Start Virtual Camera)",
                ]

        # MediaPipe instructions
        if not self.results["mediapipe"]["status"]:
            instructions["mediapipe"] = [
                f"Install MediaPipe library: pip install mediapipe",
                f"More information: {self.install_links['mediapipe']}",
            ]

            # Additional information for Python 3.11+ where MediaPipe may have issues
            if sys.version_info.major == 3 and sys.version_info.minor >= 11:
                instructions["mediapipe"].append(
                    "MediaPipe may have compatibility issues with Python 3.11+. "
                    "Consider using Python 3.9 or 3.10."
                )

        return instructions

    def _log(self, message: str, level: str = "INFO") -> None:
        """
        Logs message to logger if available.

        Args:
            message (str): Message to log
            level (str): Logging level ("INFO", "WARNING", "ERROR", "DEBUG")
        """
        if self.logger is None:
            return

        method = getattr(self.logger, level.lower(), None)
        if method:
            method("SystemCheck", message)


# Function for use in main.py
def check_system_requirements(logger=None) -> Tuple[bool, Dict[str, Any]]:
    """
    Checks if system meets application requirements.

    Args:
        logger: Optional logger for recording check results

    Returns:
        Tuple[bool, Dict[str, Any]]: (True if all requirements are met, check results)
    """
    checker = SystemCheck(logger)
    checker.check_all()

    all_requirements_met = checker.are_all_requirements_met()
    results = {
        "all_met": all_requirements_met,
        "results": checker.results,
        "missing": checker.get_missing_components(),
        "instructions": checker.get_installation_instructions(),
    }

    return all_requirements_met, results
