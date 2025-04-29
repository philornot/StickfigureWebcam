#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/camera/virtual_camera.py

import platform
import time
from typing import Optional, Dict, Any

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
            device_name: str = "Stick Figure Webcam",
            logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja wirtualnej kamery.

        Args:
            width (int): Szerokość obrazu wirtualnej kamery
            height (int): Wysokość obrazu wirtualnej kamery
            fps (int): Liczba klatek na sekundę
            device_name (str): Nazwa wyświetlana dla urządzenia
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.device_name = device_name
        self.logger = logger or CustomLogger()

        self.cam = None
        self.is_initialized = False
        self.frame_count = 0
        self.last_frame_time = 0
        self.performance = PerformanceMonitor("VirtualCamera")

        # Informacje o wirtualnej kamerze
        self.camera_info = {
            "name": device_name,
            "resolution": (width, height),
            "fps": fps,
            "backend": "unknown",
            "system": platform.system()
        }

        self.fps_sleep_time = 1.0 / fps if fps > 0 else 0

        # Na razie nie uruchamiamy automatycznie, będzie to robione przy pierwszym zapytaniu

    def initialize(self) -> bool:
        """
        Inicjalizuje wirtualną kamerę.

        Returns:
            bool: True jeśli udało się zainicjalizować kamerę, False w przeciwnym razie
        """
        if self.is_initialized:
            return True

        try:
            self.logger.debug(
                "VirtualCamera",
                f"Inicjalizacja wirtualnej kamery: {self.width}x{self.height} @ {self.fps} FPS",
                log_type="VIRTUAL_CAM"
            )

            # Określenie formatu pikseli - OpenCV używa BGR
            fmt = pyvirtualcam.PixelFormat.BGR

            # Różne systemy operacyjne mogą wymagać różnych ustawień
            system = platform.system()
            backend_params = {}

            if system == "Windows":
                # Na Windows można określić nazwę urządzenia
                backend_params["device"] = self.device_name
            elif system == "Darwin":  # macOS
                # macOS może wymagać OBS Virtual Camera plugin
                self.logger.debug(
                    "VirtualCamera",
                    "macOS wykryty - upewnij się, że OBS Virtual Camera jest zainstalowany",
                    log_type="VIRTUAL_CAM"
                )
            elif system == "Linux":
                # Linux może wymagać v4l2loopback
                self.logger.debug(
                    "VirtualCamera",
                    "Linux wykryty - upewnij się, że moduł v4l2loopback jest załadowany",
                    log_type="VIRTUAL_CAM"
                )

            # Próba utworzenia wirtualnej kamery z automatyczną detekcją backendu
            self.cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=fmt,
                **backend_params
            )

            # Aktualizacja informacji o backendzie
            self.camera_info["backend"] = self.cam.backend

            self.logger.info(
                "VirtualCamera",
                f"Wirtualna kamera uruchomiona z backendem: {self.cam.backend}",
                log_type="VIRTUAL_CAM"
            )

            self.is_initialized = True
            self.last_frame_time = time.time()

            # Powiadom logger o statusie
            self.logger.virtual_camera_status(True, self.camera_info)

            return True

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
                "jest zainstalowany i virtual camera jest uruchomiona.",
                log_type="VIRTUAL_CAM"
            )
        elif system == "Darwin":  # macOS
            self.logger.info(
                "VirtualCamera",
                "Rozwiązywanie problemów (macOS): Zainstaluj OBS Studio i plugin "
                "obs-mac-virtualcam. Alternatywnie możesz użyć CamTwist.",
                log_type="VIRTUAL_CAM"
            )
        elif system == "Linux":
            self.logger.info(
                "VirtualCamera",
                "Rozwiązywanie problemów (Linux): Upewnij się, że "
                "v4l2loopback jest zainstalowany i załadowany:\n"
                "sudo apt-get install v4l2loopback-dkms\n"
                "sudo modprobe v4l2loopback",
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
        if not self.is_initialized and not self.initialize():
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
            elapsed = current_time - self.last_frame_time

            # Jeśli potrzeba, poczekaj aby utrzymać stały FPS
            if elapsed < self.fps_sleep_time:
                time.sleep(self.fps_sleep_time - elapsed)

            self.last_frame_time = time.time()
            self.frame_count += 1

            # Co 100 klatek logujemy informacje
            if self.frame_count % 100 == 0:
                # Oblicz rzeczywisty FPS
                real_fps = 1.0 / (time.time() - current_time) if elapsed > 0 else self.fps

                self.logger.debug(
                    "VirtualCamera",
                    f"Wysłano {self.frame_count} klatek, aktualny FPS: {real_fps:.1f}",
                    log_type="VIRTUAL_CAM"
                )

            self.performance.stop_timer()
            processing_time = self.performance.get_last_execution_time() * 1000  # ms

            # Co 300 klatek logujemy informacje o wydajności
            if self.frame_count % 300 == 0:
                real_fps = 1.0 / elapsed if elapsed > 0 else self.fps
                self.logger.performance_metrics(real_fps, processing_time, "VirtualCamera")

            return True

        except Exception as e:
            self.logger.error(
                "VirtualCamera",
                f"Błąd podczas wysyłania klatki: {str(e)}",
                log_type="VIRTUAL_CAM",
                error={"error": str(e)}
            )
            return False

    def send_black_frame(self) -> bool:
        """
        Wysyła czarną klatkę do wirtualnej kamery.

        Returns:
            bool: True jeśli udało się wysłać klatkę, False w przeciwnym razie
        """
        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.send_frame(black_frame)

    def send_white_frame(self) -> bool:
        """
        Wysyła białą klatkę do wirtualnej kamery.

        Returns:
            bool: True jeśli udało się wysłać klatkę, False w przeciwnym razie
        """
        white_frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        return self.send_frame(white_frame)

    def send_test_pattern(self) -> bool:
        """
        Wysyła wzór testowy do wirtualnej kamery.

        Returns:
            bool: True jeśli udało się wysłać klatkę, False w przeciwnym razie
        """
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

        # Obramowanie tekstu dla lepszej widoczności
        cv2.putText(frame, text, (text_x - 1, text_y - 1), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x + 1, text_y + 1), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

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

        self.logger.info(
            "VirtualCamera",
            f"Ustawiono nowy FPS: {fps}. "
            "Kamera zostanie zainicjalizowana przy następnym użyciu.",
            log_type="VIRTUAL_CAM"
        )

        return True

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
