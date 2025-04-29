#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Tuple, Dict, Optional, Any, List

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class CameraCapture:
    """
    Klasa do przechwytywania obrazu z kamery internetowej.
    Zapewnia interfejs do konfiguracji kamery, przechwytywania klatek
    i podstawowych operacji na obrazie.
    """

    def __init__(
            self,
            camera_id: int = 0,
            width: int = 640,
            height: int = 480,
            fps: int = 30,
            logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja modułu przechwytywania kamery.

        Args:
            camera_id (int): Identyfikator kamery (zwykle 0 dla domyślnej kamery)
            width (int): Preferowana szerokość obrazu
            height (int): Preferowana wysokość obrazu
            fps (int): Preferowana liczba klatek na sekundę
            logger (CustomLogger, optional): Logger do zapisywania komunikatów
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger or CustomLogger()

        self.cap = None
        self.is_open = False
        self.frame_count = 0
        self.last_frame = None
        self.last_frame_time = 0
        self.performance = PerformanceMonitor("CameraCapture")

        # Informacje o kamerze
        self.camera_info = {
            "id": camera_id,
            "name": "Unknown",
            "resolution": (width, height),
            "fps": fps,
            "real_fps": 0.0
        }

        # Automatyczne otwieranie kamery przy inicjalizacji
        self.open()

    def open(self) -> bool:
        """
        Otwiera połączenie z kamerą i konfiguruje parametry.

        Returns:
            bool: True jeśli udało się otworzyć kamerę, False w przeciwnym razie
        """
        try:
            self.logger.debug("CameraCapture", f"Próba otwarcia kamery ID: {self.camera_id}", log_type="CAMERA")
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                self.logger.error("CameraCapture", f"Nie udało się otworzyć kamery ID: {self.camera_id}",
                                  log_type="CAMERA")
                self.is_open = False
                return False

            # Konfiguracja parametrów kamery
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Odczytanie rzeczywistych parametrów
            real_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            real_fps = self.cap.get(cv2.CAP_PROP_FPS)

            # Aktualizacja informacji o kamerze
            self.camera_info.update({
                "resolution": (real_width, real_height),
                "fps": real_fps,
                "backend": self.cap.getBackendName()
            })

            self.width, self.height = real_width, real_height
            self.is_open = True

            self.logger.info(
                "CameraCapture",
                f"Kamera otwarta: {self.width}x{self.height} @ {real_fps:.1f} FPS",
                log_type="CAMERA",
                camera_info=self.camera_info
            )

            # Powiadomienie loggera o statusie kamery
            self.logger.camera_status(True, self.camera_info)

            # Inicjalizacja pierwszej klatki
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.last_frame_time = time.time()
                self.frame_count = 1

            return True

        except Exception as e:
            self.is_open = False
            error_info = {"error": str(e)}
            self.logger.critical(
                "CameraCapture",
                f"Błąd podczas otwierania kamery: {str(e)}",
                log_type="CAMERA",
                error=error_info
            )
            self.logger.camera_status(False, error_info)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Odczytuje klatkę z kamery.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - bool: True jeśli udało się odczytać klatkę
                - np.ndarray: Klatka jako tablica NumPy lub None w przypadku błędu
        """
        if not self.is_open or self.cap is None:
            self.logger.warning("CameraCapture", "Próba odczytu z nieotwartej kamery", log_type="CAMERA")
            return False, None

        self.performance.start_timer()

        try:
            ret, frame = self.cap.read()

            if ret:
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_frame_time

                if elapsed > 0:
                    # Aktualizacja rzeczywistego FPS (wygładzanie wykładnicze)
                    alpha = 0.3  # Współczynnik wygładzania
                    current_fps = 1.0 / elapsed
                    if self.camera_info["real_fps"] == 0:
                        self.camera_info["real_fps"] = current_fps
                    else:
                        self.camera_info["real_fps"] = (1 - alpha) * self.camera_info["real_fps"] + alpha * current_fps

                self.last_frame = frame
                self.last_frame_time = current_time

                # Co 100 klatek logujemy statystyki wydajności
                if self.frame_count % 100 == 0:
                    self.logger.debug(
                        "CameraCapture",
                        f"Odczytano {self.frame_count} klatek, aktualny FPS: {self.camera_info['real_fps']:.1f}",
                        log_type="CAMERA"
                    )

                self.performance.stop_timer()
                processing_time = self.performance.get_last_execution_time() * 1000  # ms

                # Co 500 klatek logujemy informacje o wydajności
                if self.frame_count % 500 == 0:
                    self.logger.performance_metrics(
                        self.camera_info["real_fps"],
                        processing_time,
                        "CameraCapture"
                    )

                return True, frame
            else:
                self.logger.warning(
                    "CameraCapture",
                    "Nie udało się odczytać klatki z kamery",
                    log_type="CAMERA"
                )
                return False, None

        except Exception as e:
            self.logger.error(
                "CameraCapture",
                f"Błąd podczas odczytu z kamery: {str(e)}",
                log_type="CAMERA",
                error={"error": str(e)}
            )
            return False, None

    def get_latest_frame(self) -> np.ndarray:
        """
        Zwraca ostatnią pomyślnie odczytaną klatkę.

        Returns:
            np.ndarray: Ostatnia klatka lub pusta klatka jeśli nie ma dostępnych
        """
        if self.last_frame is not None:
            return self.last_frame.copy()
        else:
            # Zwróć pustą (czarną) klatkę
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_camera_info(self) -> Dict[str, Any]:
        """
        Zwraca informacje o kamerze.

        Returns:
            Dict[str, Any]: Słownik z informacjami o kamerze
        """
        return self.camera_info.copy()

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Ustawia rozdzielczość kamery.

        Args:
            width (int): Nowa szerokość
            height (int): Nowa wysokość

        Returns:
            bool: True jeśli udało się zmienić rozdzielczość
        """
        if not self.is_open or self.cap is None:
            self.logger.warning(
                "CameraCapture",
                "Próba zmiany rozdzielczości nieotwartej kamery",
                log_type="CAMERA"
            )
            return False

        try:
            self.logger.debug(
                "CameraCapture",
                f"Zmiana rozdzielczości na {width}x{height}",
                log_type="CAMERA"
            )

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Sprawdzenie czy udało się zmienić rozdzielczość
            real_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.width, self.height = real_width, real_height
            self.camera_info["resolution"] = (real_width, real_height)

            self.logger.info(
                "CameraCapture",
                f"Nowa rozdzielczość: {real_width}x{real_height}",
                log_type="CAMERA",
                camera_info=self.camera_info
            )

            # Jeśli rozmiary się różnią, ostrzeżenie
            if real_width != width or real_height != height:
                self.logger.warning(
                    "CameraCapture",
                    f"Żądana rozdzielczość {width}x{height} nie jest obsługiwana. "
                    f"Ustawiono najbliższą dostępną: {real_width}x{real_height}",
                    log_type="CAMERA"
                )

            return True

        except Exception as e:
            self.logger.error(
                "CameraCapture",
                f"Błąd podczas zmiany rozdzielczości: {str(e)}",
                log_type="CAMERA",
                error={"error": str(e)}
            )
            return False

    def set_fps(self, fps: int) -> bool:
        """
        Ustawia docelową liczbę klatek na sekundę.

        Args:
            fps (int): Nowa wartość FPS

        Returns:
            bool: True jeśli udało się zmienić FPS
        """
        if not self.is_open or self.cap is None:
            self.logger.warning(
                "CameraCapture",
                "Próba zmiany FPS nieotwartej kamery",
                log_type="CAMERA"
            )
            return False

        try:
            self.logger.debug("CameraCapture", f"Zmiana FPS na {fps}", log_type="CAMERA")

            self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Sprawdzenie rzeczywistego FPS
            real_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = real_fps
            self.camera_info["fps"] = real_fps

            self.logger.info(
                "CameraCapture",
                f"Nowy FPS: {real_fps:.1f}",
                log_type="CAMERA"
            )

            if abs(real_fps - fps) > 0.1:
                self.logger.warning(
                    "CameraCapture",
                    f"Żądany FPS {fps} nie jest obsługiwany. "
                    f"Ustawiono najbliższą dostępną wartość: {real_fps:.1f}",
                    log_type="CAMERA"
                )

            return True

        except Exception as e:
            self.logger.error(
                "CameraCapture",
                f"Błąd podczas zmiany FPS: {str(e)}",
                log_type="CAMERA",
                error={"error": str(e)}
            )
            return False

    def list_available_cameras(self) -> List[Dict[str, Any]]:
        """
        Wykrywa dostępne kamery w systemie.

        Returns:
            List[Dict[str, Any]]: Lista słowników z informacjami o dostępnych kamerach
        """
        available_cameras = []
        max_cameras = 10  # Ograniczenie do 10 kamer

        self.logger.debug("CameraCapture", "Wyszukiwanie dostępnych kamer...", log_type="CAMERA")

        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_info = {
                        "id": i,
                        "resolution": (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        ),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "backend": cap.getBackendName()
                    }

                    # Pobierz jedną klatkę do weryfikacji
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(camera_info)
                        self.logger.debug(
                            "CameraCapture",
                            f"Znaleziono kamerę ID: {i} - {camera_info['resolution']} @ {camera_info['fps']} FPS",
                            log_type="CAMERA"
                        )

                cap.release()

            except Exception as e:
                self.logger.trace(
                    "CameraCapture",
                    f"Błąd podczas sprawdzania kamery ID: {i}: {str(e)}",
                    log_type="CAMERA"
                )

        self.logger.info(
            "CameraCapture",
            f"Znaleziono {len(available_cameras)} dostępnych kamer",
            log_type="CAMERA",
            cameras=available_cameras
        )

        return available_cameras

    def flip_horizontal(self, frame: np.ndarray) -> np.ndarray:
        """
        Odbija klatkę w poziomie.

        Args:
            frame (np.ndarray): Klatka wejściowa

        Returns:
            np.ndarray: Odbita klatka
        """
        return cv2.flip(frame, 1)

    def adjust_brightness_contrast(
            self,
            frame: np.ndarray,
            brightness: float = 0,
            contrast: float = 1.0
    ) -> np.ndarray:
        """
        Dostosowuje jasność i kontrast klatki.

        Args:
            frame (np.ndarray): Klatka wejściowa
            brightness (float): Wartość jasności (-1.0 do 1.0)
            contrast (float): Wartość kontrastu (0.0 do 3.0)

        Returns:
            np.ndarray: Przetworzona klatka
        """
        # Konwersja jasności z zakresu -1.0:1.0 do wartości przesunięcia pikseli
        brightness_value = int(brightness * 255)

        # Zastosowanie kontrastu i jasności
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness_value)

        return adjusted

    def close(self) -> None:
        """
        Zamyka połączenie z kamerą.
        """
        if self.is_open and self.cap is not None:
            self.cap.release()
            self.is_open = False
            self.logger.info("CameraCapture", "Kamera zamknięta", log_type="CAMERA")

    def __del__(self):
        """
        Destruktor klasy, zapewniający zamknięcie kamery.
        """
        self.close()
