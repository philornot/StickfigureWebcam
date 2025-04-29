#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import subprocess
import sys
from typing import Dict, List, Tuple, Any

import cv2

# Próbujemy zaimportować pyvirtualcam, ale nie reagujemy na błąd
# (będziemy później sprawdzać czy jest dostępny)
try:
    import pyvirtualcam

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False

# Próbujemy zaimportować mediapipe
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class SystemCheck:
    """
    Klasa do sprawdzania dostępności i poprawności konfiguracji
    niezbędnych komponentów aplikacji.
    """

    def __init__(self, logger=None):
        """
        Inicjalizacja sprawdzania systemu.

        Args:
            logger: Opcjonalny logger do zapisywania wyników sprawdzeń
        """
        self.logger = logger
        self.system = platform.system()  # 'Windows', 'Linux', 'Darwin' (macOS)

        # Wyniki sprawdzeń
        self.results = {
            "camera": {"status": False, "message": "", "details": {}},
            "virtual_camera": {"status": False, "message": "", "details": {}},
            "mediapipe": {"status": False, "message": "", "details": {}},
            "obs": {"status": False, "message": "", "details": {}},
            "v4l2loopback": {"status": False, "message": "", "details": {}}
        }

        # Linki do instalacji komponentów
        self.install_links = {
            "obs": "https://obsproject.com/download",
            "v4l2loopback": "https://github.com/umlaeute/v4l2loopback",
            "pyvirtualcam": "https://pypi.org/project/pyvirtualcam/",
            "mediapipe": "https://pypi.org/project/mediapipe/",
            "obs_virtualcam_plugin_mac": "https://github.com/johnboiles/obs-mac-virtualcam"
        }

    def check_all(self) -> Dict[str, Any]:
        """
        Wykonuje wszystkie sprawdzenia systemu.

        Returns:
            Dict[str, Any]: Słownik z wynikami sprawdzeń
        """
        self._log("Rozpoczęcie sprawdzania systemu...")

        # Sprawdzenie kamery
        self.check_camera()

        # Sprawdzenie wirtualnej kamery
        self.check_virtual_camera()

        # Sprawdzenie MediaPipe
        self.check_mediapipe()

        # Sprawdzanie OBS (tylko na Windows i macOS)
        if self.system in ["Windows", "Darwin"]:
            self.check_obs()

        # Sprawdzanie v4l2loopback (tylko na Linux)
        if self.system == "Linux":
            self.check_v4l2loopback()

        self._log("Zakończenie sprawdzania systemu")

        return self.results

    def check_camera(self, camera_id: int = 0) -> Dict[str, Any]:
        """
        Sprawdza, czy kamera jest dostępna i działa poprawnie.

        Args:
            camera_id (int): Identyfikator kamery do sprawdzenia

        Returns:
            Dict[str, Any]: Wynik sprawdzenia
        """
        self._log(f"Sprawdzanie kamery (ID: {camera_id})...")

        result = self.results["camera"]
        result["details"]["camera_id"] = camera_id

        try:
            # Próba otwarcia kamery
            cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                result["status"] = False
                result["message"] = f"Nie można otworzyć kamery o ID: {camera_id}"
                self._log(result["message"], level="WARNING")
            else:
                # Sprawdzenie, czy można odczytać klatkę
                ret, frame = cap.read()

                if not ret:
                    result["status"] = False
                    result["message"] = f"Kamera o ID: {camera_id} jest dostępna, ale nie można odczytać klatki"
                    self._log(result["message"], level="WARNING")
                else:
                    # Odczytanie parametrów kamery
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    result["status"] = True
                    result["message"] = f"Kamera o ID: {camera_id} działa poprawnie"
                    result["details"].update({
                        "width": width,
                        "height": height,
                        "fps": fps
                    })
                    self._log(result["message"])

                # Zamknięcie kamery
                cap.release()
        except Exception as e:
            result["status"] = False
            result["message"] = f"Błąd podczas sprawdzania kamery: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_virtual_camera(self) -> Dict[str, Any]:
        """
        Sprawdza, czy wirtualna kamera jest dostępna i skonfigurowana.

        Returns:
            Dict[str, Any]: Wynik sprawdzenia
        """
        self._log("Sprawdzanie wirtualnej kamery...")

        result = self.results["virtual_camera"]

        # Najpierw sprawdzamy, czy pyvirtualcam jest zainstalowany
        if not PYVIRTUALCAM_AVAILABLE:
            result["status"] = False
            result["message"] = "Biblioteka pyvirtualcam nie jest zainstalowana"
            result["details"]["install_command"] = "pip install pyvirtualcam"
            result["details"]["install_link"] = self.install_links["pyvirtualcam"]
            self._log(result["message"], level="WARNING")
            return result

        try:
            # Sprawdzanie dostępnych backenów - funkcja może nie być dostępna w starszych wersjach
            available_backends = []
            try:
                # Próba użycia get_available_backends (nowsze wersje pyvirtualcam)
                if hasattr(pyvirtualcam, 'get_available_backends'):
                    available_backends = pyvirtualcam.get_available_backends()
                    result["details"]["available_backends"] = available_backends
            except Exception:
                pass

            # Próba utworzenia wirtualnej kamery
            try:
                # Używamy małej rozdzielczości dla testu
                cam = pyvirtualcam.Camera(width=320, height=240, fps=20)
                cam_info = {
                    "backend": cam.backend,
                    "width": cam.width,
                    "height": cam.height,
                    "fps": cam.fps,
                    "device": getattr(cam, 'device', None)
                }
                cam.close()

                result["status"] = True
                result["message"] = f"Wirtualna kamera działa poprawnie (backend: {cam_info['backend']})"
                result["details"].update(cam_info)
                self._log(result["message"])

            except Exception as e:
                result["status"] = False
                result["message"] = f"Błąd podczas sprawdzania wirtualnej kamery: {str(e)}"
                result["details"]["error"] = str(e)
                self._log(result["message"], level="WARNING")

                # Dodajemy sugestie w zależności od systemu
                if self.system == "Windows":
                    if "OBS" in str(e):
                        result["details"]["suggestion"] = "Zainstaluj OBS Studio i uruchom Virtual Camera"
                        result["details"]["install_link"] = self.install_links["obs"]
                elif self.system == "Linux":
                    if "v4l2loopback" in str(e):
                        result["details"]["suggestion"] = "Zainstaluj i załaduj moduł v4l2loopback"
                        result["details"]["install_link"] = self.install_links["v4l2loopback"]
                elif self.system == "Darwin":  # macOS
                    result["details"]["suggestion"] = "Zainstaluj OBS Studio i plugin obs-mac-virtualcam"
                    result["details"]["install_link_obs"] = self.install_links["obs"]
                    result["details"]["install_link_plugin"] = self.install_links["obs_virtualcam_plugin_mac"]

        except Exception as e:
            result["status"] = False
            result["message"] = f"Błąd podczas sprawdzania wirtualnej kamery: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_mediapipe(self) -> Dict[str, Any]:
        """
        Sprawdza, czy MediaPipe jest dostępny i działa poprawnie.

        Returns:
            Dict[str, Any]: Wynik sprawdzenia
        """
        self._log("Sprawdzanie MediaPipe...")

        result = self.results["mediapipe"]

        if not MEDIAPIPE_AVAILABLE:
            result["status"] = False
            result["message"] = "Biblioteka MediaPipe nie jest zainstalowana"
            result["details"]["install_command"] = "pip install mediapipe"
            result["details"]["install_link"] = self.install_links["mediapipe"]
            self._log(result["message"], level="WARNING")
            return result

        try:
            # Próba utworzenia detektora pozy
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,  # Używamy najprostszego modelu dla testu
                min_detection_confidence=0.5
            )

            # Sprawdzenie wersji MediaPipe
            mp_version = mp.__version__

            result["status"] = True
            result["message"] = f"MediaPipe działa poprawnie (wersja: {mp_version})"
            result["details"]["version"] = mp_version
            self._log(result["message"])

            # Zamknięcie detektora
            pose.close()

        except Exception as e:
            result["status"] = False
            result["message"] = f"Błąd podczas sprawdzania MediaPipe: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_obs(self) -> Dict[str, Any]:
        """
        Sprawdza, czy OBS Studio jest zainstalowany (tylko Windows/macOS).

        Returns:
            Dict[str, Any]: Wynik sprawdzenia
        """
        self._log("Sprawdzanie OBS Studio...")

        result = self.results["obs"]

        # Sprawdzamy tylko na Windows i macOS
        if self.system not in ["Windows", "Darwin"]:
            result["status"] = None
            result["message"] = "Sprawdzanie OBS pominięte - nieobsługiwany system"
            return result

        try:
            # Ścieżki instalacyjne OBS
            obs_paths = []

            if self.system == "Windows":
                # Typowe lokalizacje na Windows
                program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
                program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")

                obs_paths = [
                    os.path.join(program_files, "obs-studio"),
                    os.path.join(program_files_x86, "obs-studio")
                ]

                # Dla Steam
                steam_path = os.path.join(program_files, "Steam", "steamapps", "common", "obs-studio")
                if os.path.exists(steam_path):
                    obs_paths.append(steam_path)

            elif self.system == "Darwin":  # macOS
                # Typowe lokalizacje na macOS
                obs_paths = [
                    "/Applications/OBS.app",
                    os.path.expanduser("~/Applications/OBS.app")
                ]

            # Sprawdzenie, czy OBS istnieje w którejś z lokalizacji
            obs_installed = False
            obs_location = None

            for path in obs_paths:
                if os.path.exists(path):
                    obs_installed = True
                    obs_location = path
                    break

            if obs_installed:
                result["status"] = True
                result["message"] = "OBS Studio jest zainstalowany"
                result["details"]["location"] = obs_location
                self._log(result["message"])
            else:
                result["status"] = False
                result["message"] = "OBS Studio nie jest zainstalowany"
                result["details"]["install_link"] = self.install_links["obs"]
                self._log(result["message"], level="WARNING")

        except Exception as e:
            result["status"] = None
            result["message"] = f"Błąd podczas sprawdzania OBS: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def check_v4l2loopback(self) -> Dict[str, Any]:
        """
        Sprawdza, czy moduł v4l2loopback jest zainstalowany i załadowany (tylko Linux).

        Returns:
            Dict[str, Any]: Wynik sprawdzenia
        """
        self._log("Sprawdzanie modułu v4l2loopback...")

        result = self.results["v4l2loopback"]

        # Sprawdzamy tylko na Linuxie
        if self.system != "Linux":
            result["status"] = None
            result["message"] = "Sprawdzanie v4l2loopback pominięte - nieobsługiwany system"
            return result

        try:
            # Sprawdzenie, czy moduł jest załadowany
            module_loaded = False
            loaded_modules = subprocess.check_output(['lsmod']).decode('utf-8')

            if 'v4l2loopback' in loaded_modules:
                module_loaded = True

            # Sprawdzenie, czy istnieją urządzenia wirtualnej kamery
            v4l_devices = []

            try:
                devices_output = subprocess.check_output(['ls', '-la', '/dev/video*']).decode('utf-8')
                v4l_devices = [line for line in devices_output.splitlines() if 'video' in line]
            except subprocess.CalledProcessError:
                pass

            if module_loaded:
                result["status"] = True
                result["message"] = "Moduł v4l2loopback jest załadowany"
                result["details"]["devices"] = len(v4l_devices)
                self._log(result["message"])
            else:
                result["status"] = False
                result["message"] = "Moduł v4l2loopback nie jest załadowany"
                result["details"]["install_command"] = "sudo apt-get install v4l2loopback-dkms"
                result["details"]["load_command"] = "sudo modprobe v4l2loopback"
                result["details"]["install_link"] = self.install_links["v4l2loopback"]
                self._log(result["message"], level="WARNING")

        except Exception as e:
            result["status"] = None
            result["message"] = f"Błąd podczas sprawdzania v4l2loopback: {str(e)}"
            result["details"]["error"] = str(e)
            self._log(result["message"], level="ERROR")

        return result

    def get_missing_components(self) -> List[Dict[str, Any]]:
        """
        Zwraca listę brakujących lub niepoprawnie skonfigurowanych komponentów.

        Returns:
            List[Dict[str, Any]]: Lista brakujących komponentów z informacjami
        """
        missing = []

        for component, result in self.results.items():
            if result["status"] is False:  # Pomijamy None (nieobsługiwane) i True (OK)
                missing.append({
                    "name": component,
                    "message": result["message"],
                    "details": result["details"]
                })

        return missing

    def are_all_requirements_met(self) -> bool:
        """
        Sprawdza, czy wszystkie wymagane komponenty są dostępne.

        Returns:
            bool: True jeśli wszystkie wymagania są spełnione
        """
        # Sprawdzamy tylko komponenty wymagane dla danego systemu
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
        Generuje instrukcje instalacji dla brakujących komponentów.

        Returns:
            Dict[str, List[str]]: Słownik instrukcji dla różnych komponentów
        """
        instructions = {}

        # Instrukcje dla kamery
        if not self.results["camera"]["status"]:
            instructions["camera"] = [
                "Sprawdź, czy kamera jest podłączona i działa poprawnie.",
                "Upewnij się, że żadna inna aplikacja nie używa kamery.",
                "Sprawdź ustawienia prywatności systemu i uprawnienia aplikacji do korzystania z kamery."
            ]

        # Instrukcje dla wirtualnej kamery
        if not self.results["virtual_camera"]["status"]:
            if self.system == "Windows":
                instructions["virtual_camera"] = [
                    f"Zainstaluj OBS Studio: {self.install_links['obs']}",
                    "Uruchom OBS Studio i włącz Virtual Camera (Narzędzia -> Start Virtual Camera)",
                    "Jeśli już masz OBS, upewnij się, że Virtual Camera jest włączona"
                ]
            elif self.system == "Linux":
                instructions["virtual_camera"] = [
                    f"Zainstaluj v4l2loopback: {self.install_links['v4l2loopback']}",
                    "Instalacja przez apt: sudo apt-get install v4l2loopback-dkms",
                    "Załaduj moduł: sudo modprobe v4l2loopback"
                ]
            elif self.system == "Darwin":  # macOS
                instructions["virtual_camera"] = [
                    f"Zainstaluj OBS Studio: {self.install_links['obs']}",
                    f"Zainstaluj plugin obs-mac-virtualcam: {self.install_links['obs_virtualcam_plugin_mac']}",
                    "Uruchom OBS Studio i włącz Virtual Camera (Narzędzia -> Start Virtual Camera)"
                ]

        # Instrukcje dla MediaPipe
        if not self.results["mediapipe"]["status"]:
            instructions["mediapipe"] = [
                f"Zainstaluj bibliotekę MediaPipe: pip install mediapipe",
                f"Więcej informacji: {self.install_links['mediapipe']}"
            ]

            # Dodatkowe informacje dla Python 3.11+ gdzie MediaPipe może mieć problemy
            if sys.version_info.major == 3 and sys.version_info.minor >= 11:
                instructions["mediapipe"].append(
                    "MediaPipe może mieć problemy z kompatybilnością z Python 3.11+. "
                    "Rozważ użycie Python 3.9 lub 3.10."
                )

        return instructions

    def _log(self, message: str, level: str = "INFO") -> None:
        """
        Zapisuje komunikat w loggerze, jeśli jest dostępny.

        Args:
            message (str): Komunikat do zapisania
            level (str): Poziom logowania ("INFO", "WARNING", "ERROR", "DEBUG")
        """
        if self.logger is None:
            return

        method = getattr(self.logger, level.lower(), None)
        if method:
            method("SystemCheck", message)


# Funkcja do wykorzystania w main.py
def check_system_requirements(logger=None) -> Tuple[bool, Dict[str, Any]]:
    """
    Sprawdza, czy system spełnia wymagania aplikacji.

    Args:
        logger: Opcjonalny logger do zapisywania wyników sprawdzeń

    Returns:
        Tuple[bool, Dict[str, Any]]: (True jeśli wszystkie wymagania są spełnione, wyniki sprawdzeń)
    """
    checker = SystemCheck(logger)
    checker.check_all()

    all_requirements_met = checker.are_all_requirements_met()
    results = {
        "all_met": all_requirements_met,
        "results": checker.results,
        "missing": checker.get_missing_components(),
        "instructions": checker.get_installation_instructions()
    }

    return all_requirements_met, results
