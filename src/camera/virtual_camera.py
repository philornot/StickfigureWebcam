#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/camera/virtual_camera.py

import platform
import time
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import pyvirtualcam

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class VirtualCamera:
    """
    Klasa do tworzenia wirtualnej kamery dostępnej dla innych aplikacji.
    Umożliwia przekazywanie wygenerowanego obrazu jako źródła wideo.
    """

    def __init__(
            self,
            width: int = 640,
            height: int = 480,
            fps: int = 30,
            device_name: str = None,  # Domyślnie None dla auto-wykrywania
            logger: Optional[CustomLogger] = None,
            max_retries: int = 3,  # Maksymalna liczba prób
            retry_delay: float = 1.0  # Opóźnienie między próbami
    ):
        """
        Inicjalizacja wirtualnej kamery.

        Args:
            width (int): Szerokość obrazu wirtualnej kamery
            height (int): Wysokość obrazu wirtualnej kamery
            fps (int): Liczba klatek na sekundę
            device_name (str, optional): Nazwa wyświetlana dla urządzenia (domyślnie: auto-wykrywana)
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
            max_retries (int): Maksymalna liczba prób inicjalizacji
            retry_delay (float): Opóźnienie między próbami w sekundach
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.device_name = device_name
        self.logger = logger or CustomLogger()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.cam = None
        self.is_initialized = False
        self.initialization_failed = False  # Flaga oznaczająca definitywne niepowodzenie
        self.frame_count = 0
        self.last_frame_time = 0
        self.performance = PerformanceMonitor("VirtualCamera")
        self.retry_count = 0  # Licznik prób inicjalizacji

        # Informacje o wirtualnej kamerze
        self.camera_info = {
            "name": device_name if device_name else "Auto-detected",
            "resolution": (width, height),
            "fps": fps,
            "backend": "unknown",
            "system": platform.system(),
            "available_devices": []
        }

        self.fps_sleep_time = 1.0 / fps if fps > 0 else 0

        # Automatycznie wykrywamy dostępne urządzenia
        self._detect_available_devices()

    def _detect_available_devices(self) -> None:
        """
        Wykrywa dostępne urządzenia wirtualnej kamery.
        """
        try:
            # Próba pobrania dostępnych backenów
            available_backends = []
            try:
                if hasattr(pyvirtualcam, 'get_available_backends'):
                    available_backends = pyvirtualcam.get_available_backends()
                    self.logger.debug(
                        "VirtualCamera",
                        f"Dostępne backendy: {available_backends}",
                        log_type="VIRTUAL_CAM"
                    )
            except Exception as e:
                self.logger.warning(
                    "VirtualCamera",
                    f"Błąd podczas pobierania dostępnych backenów: {str(e)}",
                    log_type="VIRTUAL_CAM"
                )

            # Próba automatycznego wykrycia nazw urządzeń
            system = platform.system()
            self.camera_info["available_devices"] = self._get_system_specific_device_names(system)

        except Exception as e:
            self.logger.warning(
                "VirtualCamera",
                f"Błąd podczas wykrywania dostępnych urządzeń: {str(e)}",
                log_type="VIRTUAL_CAM"
            )

    def _get_system_specific_device_names(self, system: str) -> List[str]:
        """
        Zwraca listę możliwych nazw urządzeń wirtualnej kamery dla danego systemu.

        Args:
            system (str): Nazwa systemu operacyjnego

        Returns:
            List[str]: Lista możliwych nazw urządzeń
        """
        if system == "Windows":
            return [
                "OBS Virtual Camera",
                "OBS Camera",
                "OBS-Camera",
                "Unity Video Capture",
                "XSplit VCam",
                "e2eSoft VCam",
                "Stick Figure Webcam"
            ]
        elif system == "Darwin":  # macOS
            return [
                "OBS Virtual Camera",
                "NDI Video",
                "CamTwist",
                "Stick Figure Webcam"
            ]
        elif system == "Linux":
            return [
                "/dev/video0",
                "/dev/video1",
                "/dev/video2",
                "/dev/video20",
                "Stick Figure Webcam"
            ]
        else:
            return ["Stick Figure Webcam"]

    def initialize(self) -> bool:
        """
        Inicjalizuje wirtualną kamerę.

        Returns:
            bool: True jeśli udało się zainicjalizować kamerę, False w przeciwnym razie
        """
        if self.is_initialized:
            return True

        # Jeśli inicjalizacja już definitywnie się nie powiodła i wyczerpano próby,
        # nie próbujemy ponownie
        if self.initialization_failed:
            return False

        try:
            self.retry_count += 1
            self.logger.debug(
                "VirtualCamera",
                f"Inicjalizacja wirtualnej kamery (próba {self.retry_count}/{self.max_retries}): "
                f"{self.width}x{self.height} @ {self.fps} FPS",
                log_type="VIRTUAL_CAM"
            )

            # Określenie formatu pikseli - OpenCV używa BGR
            fmt = pyvirtualcam.PixelFormat.BGR

            # Różne systemy operacyjne mogą wymagać różnych ustawień
            system = platform.system()
            backend_params = {}

            # Wybieramy nazwę urządzenia:
            # 1. Priorytet ma nazwa podana przez użytkownika
            # 2. Używamy domyślnej nazwy dla danego systemu
            if self.device_name:
                if system == "Windows":
                    backend_params["device"] = self.device_name
            else:
                # Dla Windows próbujemy najpierw OBS Virtual Camera
                if system == "Windows":
                    backend_params["device"] = "OBS Virtual Camera"

            # Próba utworzenia wirtualnej kamery z automatyczną detekcją backendu
            try:
                self.cam = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    fmt=fmt,
                    **backend_params
                )

                # Aktualizacja informacji o backendzie i rzeczywistej nazwie urządzenia
                self.camera_info["backend"] = self.cam.backend
                if hasattr(self.cam, 'device'):
                    self.camera_info["name"] = self.cam.device

                self.is_initialized = True
                self.retry_count = 0  # Resetujemy licznik prób

                self.logger.info(
                    "VirtualCamera",
                    f"Wirtualna kamera uruchomiona z backendem: {self.cam.backend}, "
                    f"urządzenie: {getattr(self.cam, 'device', 'default')}",
                    log_type="VIRTUAL_CAM"
                )

                self.last_frame_time = time.time()

                # Powiadom logger o statusie
                self.logger.virtual_camera_status(True, self.camera_info)

                return True

            except Exception as first_e:
                # Jeśli pierwsza próba się nie powiedzie, próbujemy z alternatywnymi nazwami urządzeń
                if system == "Windows" and self.retry_count <= 1:
                    device_names = self._get_system_specific_device_names(system)

                    for device_name in device_names:
                        try:
                            self.logger.debug(
                                "VirtualCamera",
                                f"Próba inicjalizacji z alternatywną nazwą urządzenia: {device_name}",
                                log_type="VIRTUAL_CAM"
                            )

                            self.cam = pyvirtualcam.Camera(
                                width=self.width,
                                height=self.height,
                                fps=self.fps,
                                fmt=fmt,
                                device=device_name
                            )

                            # Sukces - zapisujemy informacje
                            self.camera_info["backend"] = self.cam.backend
                            self.camera_info["name"] = device_name
                            self.is_initialized = True
                            self.retry_count = 0

                            self.logger.info(
                                "VirtualCamera",
                                f"Wirtualna kamera uruchomiona z alternatywną nazwą: {device_name}, "
                                f"backend: {self.cam.backend}",
                                log_type="VIRTUAL_CAM"
                            )

                            self.last_frame_time = time.time()
                            self.logger.virtual_camera_status(True, self.camera_info)
                            return True

                        except Exception as e:
                            # Kontynuujemy próby z innymi nazwami
                            self.logger.debug(
                                "VirtualCamera",
                                f"Próba z urządzeniem {device_name} nie powiodła się: {str(e)}",
                                log_type="VIRTUAL_CAM"
                            )

                    # Żadna alternatywna nazwa nie zadziałała, zgłaszamy oryginalny błąd
                    raise first_e
                else:
                    # Dla innych systemów lub po wyczerpaniu alternatyw, zgłaszamy oryginalny błąd
                    raise first_e

        except Exception as e:
            error_info = {"error": str(e)}
            self.logger.error(
                "VirtualCamera",
                f"Błąd podczas inicjalizacji wirtualnej kamery: {str(e)}",
                log_type="VIRTUAL_CAM",
                error=error_info
            )
            self.logger.virtual_camera_status(False, error_info)

            # Dodatkowe porady dotyczące rozwiązywania problemów
            self._provide_troubleshooting_info()

            # Sprawdzamy czy wyczerpaliśmy wszystkie próby
            if self.retry_count >= self.max_retries:
                self.initialization_failed = True
                self.logger.warning(
                    "VirtualCamera",
                    f"Wyczerpano limit {self.max_retries} prób inicjalizacji wirtualnej kamery. "
                    "Rezygnuję z dalszych prób.",
                    log_type="VIRTUAL_CAM"
                )
            else:
                # Czekamy przed kolejną próbą
                time.sleep(self.retry_delay)

            return False

    def _provide_troubleshooting_info(self) -> None:
        """
        Dostarcza informacje pomocne w rozwiązywaniu problemów z wirtualną kamerą.
        """
        system = platform.system()

        if system == "Windows":
            self.logger.info(
                "VirtualCamera",
                "Rozwiązywanie problemów (Windows): Upewnij się, że OBS Studio "
                "jest zainstalowany i virtual camera jest uruchomiona. "
                "1. Otwórz OBS Studio. "
                "2. W menu wybierz Narzędzia -> Virtual Camera. "
                "3. Kliknij 'Start Virtual Camera'.",
                log_type="VIRTUAL_CAM"
            )
        elif system == "Darwin":  # macOS
            self.logger.info(
                "VirtualCamera",
                "Rozwiązywanie problemów (macOS): Zainstaluj OBS Studio i plugin "
                "obs-mac-virtualcam. Alternatywnie możesz użyć CamTwist. "
                "Szczegółowe instrukcje: "
                "1. Zainstaluj OBS ze strony https://obsproject.com "
                "2. Zainstaluj plugin obs-mac-virtualcam "
                "3. Uruchom OBS i włącz Virtual Camera",
                log_type="VIRTUAL_CAM"
            )
        elif system == "Linux":
            self.logger.info(
                "VirtualCamera",
                "Rozwiązywanie problemów (Linux): Upewnij się, że "
                "v4l2loopback jest zainstalowany i załadowany:\n"
                "sudo apt-get install v4l2loopback-dkms\n"
                "sudo modprobe v4l2loopback\n"
                "Po załadowaniu modułu, sprawdź dostępne urządzenia: ls -l /dev/video*",
                log_type="VIRTUAL_CAM"
            )

    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Wysyła klatkę do wirtualnej kamery.

        Args:
            frame (np.ndarray): Klatka (obraz) do wysłania

        Returns:
            bool: True jeśli udało się wysłać klatkę, False w przeciwnym razie
        """
        # Jeśli inicjalizacja definitywnie się nie powiodła, nie próbuj ponownie
        if self.initialization_failed:
            return False

        # Próba inicjalizacji, ale tylko jeśli jeszcze nie było definitywnego niepowodzenia
        if not self.is_initialized and not self.initialize():
            # Zwracamy False, ale nie zgłaszamy błędu - już to zrobiliśmy w initialize()
            return False

        if self.cam is None:
            self.logger.warning(
                "VirtualCamera",
                "Próba wysłania klatki do niezainicjalizowanej kamery",
                log_type="VIRTUAL_CAM"
            )
            return False

        self.performance.start_timer()

        try:
            # Sprawdź czy rozmiar ramki jest zgodny z ustawieniami kamery
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))

            # Wysłanie klatki
            self.cam.send(frame)

            # Obliczanie czasu do następnej klatki, aby utrzymać stały FPS
            current_time = time.time()

            # Zabezpieczenie przed dzieleniem przez zero - używamy epsilon
            elapsed = max(0.000001, current_time - self.last_frame_time)  # Minimalna wartość 1µs

            # Jeśli potrzeba, poczekaj aby utrzymać stały FPS
            if self.fps_sleep_time > 0 and elapsed < self.fps_sleep_time:
                time.sleep(self.fps_sleep_time - elapsed)

            self.last_frame_time = time.time()
            self.frame_count += 1

            # Co 100 klatek logujemy informacje
            if self.frame_count % 100 == 0:
                # Oblicz rzeczywisty FPS (z zabezpieczeniem przed dzieleniem przez zero)
                # Używamy wartości self.fps jeśli elapsed jest zbyt małe
                real_fps = 1.0 / elapsed if elapsed > 0.001 else self.fps

                self.logger.debug(
                    "VirtualCamera",
                    f"Wysłano {self.frame_count} klatek, aktualny FPS: {real_fps:.1f}",
                    log_type="VIRTUAL_CAM"
                )

            self.performance.stop_timer()
            processing_time = self.performance.get_last_execution_time() * 1000  # ms

            # Co 300 klatek logujemy informacje o wydajności
            if self.frame_count % 300 == 0:
                # Zabezpieczenie przed dzieleniem przez zero
                real_fps = 1.0 / max(0.001, elapsed)
                self.logger.performance_metrics(real_fps, processing_time, "VirtualCamera")

            return True

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "VirtualCamera",
                f"Błąd podczas wysyłania klatki: {str(e)}",
                log_type="VIRTUAL_CAM",
                error={"error": str(e)}
            )

            # Resetujemy stan kamery, aby spróbować ponownej inicjalizacji
            self.is_initialized = False
            if self.cam is not None:
                try:
                    self.cam.close()
                except:
                    pass
                self.cam = None

            return False

    def send_black_frame(self) -> bool:
        """
        Wysyła czarną klatkę do wirtualnej kamery.

        Returns:
            bool: True jeśli udało się wysłać klatkę, False w przeciwnym razie
        """
        if self.initialization_failed:
            return False

        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.send_frame(black_frame)

    def send_white_frame(self) -> bool:
        """
        Wysyła białą klatkę do wirtualnej kamery.

        Returns:
            bool: True jeśli udało się wysłać klatkę, False w przeciwnym razie
        """
        if self.initialization_failed:
            return False

        white_frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        return self.send_frame(white_frame)

    def send_test_pattern(self) -> bool:
        """
        Wysyła wzór testowy do wirtualnej kamery.

        Returns:
            bool: True jeśli udało się wysłać klatkę, False w przeciwnym razie
        """
        if self.initialization_failed:
            return False

        # Utworzenie wzoru testowego - kolorowe paski, siatka, tekst
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Kolorowe paski poziome
        bar_height = self.height // 7
        colors = [
            (255, 0, 0),  # Niebieski (BGR)
            (0, 255, 0),  # Zielony
            (0, 0, 255),  # Czerwony
            (255, 255, 0),  # Turkusowy
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Żółty
            (255, 255, 255)  # Biały
        ]

        for i, color in enumerate(colors):
            start_y = i * bar_height
            end_y = (i + 1) * bar_height if i < len(colors) - 1 else self.height
            frame[start_y:end_y, :] = color

        # Dodajemy siatke
        grid_step = 50
        grid_color = (128, 128, 128)
        grid_thickness = 1

        for x in range(0, self.width, grid_step):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, grid_thickness)
        for y in range(0, self.height, grid_step):
            cv2.line(frame, (0, y), (self.width, y), grid_color, grid_thickness)

        # Dodajemy tekst
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Stick Figure Webcam {self.width}x{self.height} @ {self.fps}FPS"
        text_size, _ = cv2.getTextSize(text, font, 1, 2)
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2

        # Status inicjalizacji - czerwony jeśli nie działa, zielony jeśli działa
        status_color = (0, 255, 0) if self.is_initialized else (0, 0, 255)  # Zielony/Czerwony
        status_text = "VIRTUAL CAMERA READY" if self.is_initialized else "VIRTUAL CAMERA NOT WORKING"
        status_text_size, _ = cv2.getTextSize(status_text, font, 0.8, 2)
        status_x = (self.width - status_text_size[0]) // 2
        status_y = (self.height + status_text_size[1]) // 2 + 30

        # Obramowanie tekstu dla lepszej widoczności
        cv2.putText(frame, text, (text_x - 1, text_y - 1), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x + 1, text_y + 1), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

        # Status
        cv2.putText(frame, status_text, (status_x, status_y), font, 0.8, status_color, 2)

        return self.send_frame(frame)

    def get_camera_info(self) -> Dict[str, Any]:
        """
        Zwraca informacje o wirtualnej kamerze.

        Returns:
            Dict[str, Any]: Słownik z informacjami o kamerze
        """
        return self.camera_info.copy()

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Zmienia rozdzielczość wirtualnej kamery.
        Wymaga ponownej inicjalizacji kamery!

        Args:
            width (int): Nowa szerokość
            height (int): Nowa wysokość

        Returns:
            bool: True jeśli udało się zmienić rozdzielczość
        """
        if self.is_initialized:
            self.logger.warning(
                "VirtualCamera",
                "Zmiana rozdzielczości wymaga ponownej inicjalizacji kamery. "
                "Zamykanie aktualnej instancji...",
                log_type="VIRTUAL_CAM"
            )
            self.close()

        self.width = width
        self.height = height
        self.camera_info["resolution"] = (width, height)

        # Resetujemy flagę niepowodzenia inicjalizacji
        self.initialization_failed = False
        self.retry_count = 0

        self.logger.info(
            "VirtualCamera",
            f"Ustawiono nową rozdzielczość: {width}x{height}. "
            "Kamera zostanie zainicjalizowana przy następnym użyciu.",
            log_type="VIRTUAL_CAM"
        )

        return True

    def set_fps(self, fps: int) -> bool:
        """
        Zmienia docelowy FPS wirtualnej kamery.
        Wymaga ponownej inicjalizacji kamery!

        Args:
            fps (int): Nowy FPS

        Returns:
            bool: True jeśli udało się zmienić FPS
        """
        if self.is_initialized:
            self.logger.warning(
                "VirtualCamera",
                "Zmiana FPS wymaga ponownej inicjalizacji kamery. "
                "Zamykanie aktualnej instancji...",
                log_type="VIRTUAL_CAM"
            )
            self.close()

        self.fps = fps
        self.camera_info["fps"] = fps
        self.fps_sleep_time = 1.0 / fps if fps > 0 else 0

        # Resetujemy flagę niepowodzenia inicjalizacji
        self.initialization_failed = False
        self.retry_count = 0

        self.logger.info(
            "VirtualCamera",
            f"Ustawiono nowy FPS: {fps}. "
            "Kamera zostanie zainicjalizowana przy następnym użyciu.",
            log_type="VIRTUAL_CAM"
        )

        return True

    def reset(self) -> bool:
        """
        Resetuje wirtualną kamerę, zamykając bieżącą instancję i resetując flagi stanu.

        Returns:
            bool: True jeśli reset się powiódł
        """
        try:
            self.close()
            self.is_initialized = False
            self.initialization_failed = False
            self.retry_count = 0

            self.logger.info(
                "VirtualCamera",
                "Wirtualna kamera została zresetowana",
                log_type="VIRTUAL_CAM"
            )

            return True
        except Exception as e:
            self.logger.error(
                "VirtualCamera",
                f"Błąd podczas resetowania wirtualnej kamery: {str(e)}",
                log_type="VIRTUAL_CAM"
            )
            return False

    def close(self) -> None:
        """
        Zamyka wirtualną kamerę.
        """
        if self.is_initialized and self.cam is not None:
            try:
                # Przed zamknięciem, wyślij czarną klatkę
                self.send_black_frame()

                # Niektóre backendy mogą potrzebować krótkiej pauzy
                time.sleep(0.1)

                # Zamknij kamerę
                self.cam.close()
                self.cam = None
                self.is_initialized = False

                self.logger.info(
                    "VirtualCamera",
                    "Wirtualna kamera zamknięta",
                    log_type="VIRTUAL_CAM"
                )

            except Exception as e:
                self.logger.error(
                    "VirtualCamera",
                    f"Błąd podczas zamykania wirtualnej kamery: {str(e)}",
                    log_type="VIRTUAL_CAM",
                    error={"error": str(e)}
                )

    def __del__(self):
        """
        Destruktor klasy, zapewniający zamknięcie kamery.
        """
        self.close()
