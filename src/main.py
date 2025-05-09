#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/main.py

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np

from src.camera.camera_capture import CameraCapture
from src.camera.virtual_camera import VirtualCamera
from src.drawing.stick_figure import StickFigureRenderer
from src.lighting.adaptive_colors import AdaptiveLightingManager
from src.pose.pose_detector import PoseDetector
from src.pose.posture_analyzer import PostureAnalyzer
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor
from src.utils.setup_dialog import show_setup_dialog
from src.utils.system_check import check_system_requirements


class StickFigureWebcam:
    """
    Główna klasa aplikacji, która łączy wszystkie komponenty i zarządza przepływem danych.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        debug: bool = False,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        show_preview: bool = True,
        flip_camera: bool = True,
        adaptive_lighting: bool = False,  # Dodano parametr
        adaptation_speed: float = 0.01,  # Dodano parametr szybkości adaptacji
        logger: Optional[CustomLogger] = None
    ):
        """
        Inicjalizacja aplikacji Stick Figure Webcam.

        Args:
            camera_id (int): Identyfikator kamery (0 dla domyślnej)
            width (int): Szerokość obrazu
            height (int): Wysokość obrazu
            fps (int): Docelowa liczba klatek na sekundę
            debug (bool): Czy włączyć tryb debugowania
            log_level (str): Poziom logowania ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            log_file (Optional[str]): Ścieżka do pliku logów (None dla logowania tylko do konsoli)
            show_preview (bool): Czy pokazywać podgląd obrazu
            flip_camera (bool): Czy odbijać obraz z kamery w poziomie
            adaptive_lighting (bool): Czy włączyć adaptacyjne dostosowywanie kolorów do oświetlenia otoczenia
            adaptation_speed (float): Szybkość adaptacji kolorów (0.001-0.1)
            logger (Optional[CustomLogger]): Opcjonalny logger (zamiast tworzenia nowego)
        """
        # Konfiguracja
        self.width = width
        self.height = height
        self.fps = fps
        self.debug = debug
        self.show_preview = show_preview
        self.flip_camera = flip_camera
        self.adaptive_lighting = adaptive_lighting

        # Flagi stanu
        self.running = False
        self.paused = False

        # Użycie istniejącego loggera lub utworzenie nowego
        if logger:
            self.logger = logger
        else:
            # Ścieżka logów
            if log_file is None and not os.path.exists("logs"):
                os.makedirs("logs", exist_ok=True)
                log_file = os.path.join("logs", f"stick_figure_webcam_{time.strftime('%Y%m%d_%H%M%S')}.log")

            # Inicjalizacja loggera
            console_level = "DEBUG" if debug else log_level
            file_level = "DEBUG"  # Zawsze zapisujemy debugi do pliku

            self.logger = CustomLogger(
                log_file=log_file,
                console_level=console_level,
                file_level=file_level,
                verbose=debug
            )

        self.logger.info("Main", "Inicjalizacja Stick Figure Webcam", log_type="CONFIG")

        # Monitor wydajności
        self.performance = PerformanceMonitor("Main")

        # Inicjalizacja komponentów
        try:
            # Inicjalizacja kamery
            self.logger.info("Main", "Inicjalizacja kamery", log_type="CAMERA")
            self.camera = CameraCapture(
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps,
                logger=self.logger
            )

            # Inicjalizacja wirtualnej kamery
            self.logger.info("Main", "Inicjalizacja wirtualnej kamery", log_type="VIRTUAL_CAM")
            self.virtual_camera = VirtualCamera(
                width=width,
                height=height,
                fps=fps,
                device_name=None,  # Automatyczne wykrywanie
                logger=self.logger,
                max_retries=3
            )

            # Inicjalizacja detektora pozy
            self.logger.info("Main", "Inicjalizacja detektora pozy", log_type="POSE")
            self.pose_detector = PoseDetector(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                model_complexity=1,
                smooth_landmarks=True,
                logger=self.logger
            )

            # Inicjalizacja analizatora postawy
            self.logger.info("Main", "Inicjalizacja analizatora postawy", log_type="POSE")
            self.posture_analyzer = PostureAnalyzer(
                standing_hip_threshold=0.7,
                confidence_threshold=0.6,
                smoothing_factor=0.7,
                temporal_smoothing=5,
                partial_visibility_bias=0.8,
                logger=self.logger
            )

            # Inicjalizacja renderera stick figure
            self.logger.info("Main", "Inicjalizacja renderera stick figure", log_type="DRAWING")
            self.stick_figure_renderer = StickFigureRenderer(
                canvas_width=width,
                canvas_height=height,
                line_thickness=3,
                head_radius_factor=0.075,
                smooth_factor=0.3,
                smoothing_history=3,
                logger=self.logger
            )

            # Inicjalizacja managera adaptacyjnego oświetlenia
            if self.adaptive_lighting:
                self.logger.info("Main", "Inicjalizacja managera adaptacyjnego oświetlenia", log_type="LIGHTING")
                self.lighting_manager = AdaptiveLightingManager(
                    adaptation_speed=adaptation_speed,
                    smoothing_window=int(fps),  # 1 sekunda jako okno wygładzania
                    sampling_interval=3,  # Analizuj co 3 klatkę dla oszczędności CPU
                    logger=self.logger
                )
            else:
                self.lighting_manager = None

            # Liczniki i statystyki
            self.frame_count = 0
            self.fps_counter = 0
            self.fps_timer = time.time()
            self.current_fps = 0.0

            self.logger.info(
                "Main",
                f"Stick Figure Webcam zainicjalizowany pomyślnie ({width}x{height} @ {fps}FPS)",
                log_type="CONFIG"
            )

        except Exception as e:
            self.logger.critical(
                "Main",
                f"Błąd podczas inicjalizacji: {str(e)}",
                log_type="CONFIG",
                error={"error": str(e)}
            )
            raise

    def run(self) -> None:
        """
        Uruchamia główną pętlę aplikacji.
        """
        self.running = True
        self.logger.info("Main", "Uruchamianie głównej pętli aplikacji", log_type="CONFIG")

        # Inicjalizacja wirtualnej kamery - ale nie zatrzymujemy się,
        # jeśli nie uda się zainicjalizować
        virtual_camera_working = self.virtual_camera.initialize()
        if virtual_camera_working:
            # Wysyłamy testowy wzór tylko jeśli kamera działa
            self.virtual_camera.send_test_pattern()
        else:
            self.logger.warning(
                "Main",
                "Wirtualna kamera nie działa. Aplikacja będzie kontynuować, "
                "ale obraz nie będzie dostępny jako źródło wideo.",
                log_type="CONFIG"
            )

        try:
            # Główna pętla aplikacji
            while self.running:
                self.performance.start_timer()

                # 1. Pobieranie klatki z kamery
                ret, frame = self.camera.read()

                if not ret:
                    self.logger.warning("Main", "Nie udało się odczytać klatki z kamery", log_type="CAMERA")
                    time.sleep(0.1)  # Krótka pauza przed ponowną próbą
                    continue

                # Odbicie obrazu w poziomie jeśli potrzeba
                if self.flip_camera:
                    frame = self.camera.flip_horizontal(frame)

                # Aktualizacja licznika FPS
                self.frame_count += 1
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_timer >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_timer)
                    self.fps_timer = current_time
                    self.fps_counter = 0

                if self.paused:
                    # W trybie pauzy pokazujemy ostatni obraz lub wzór testowy
                    if virtual_camera_working:
                        self.virtual_camera.send_test_pattern()
                    if self.show_preview:
                        self._show_preview(frame, None, None)

                    # Dodajemy obsługę klawiszy w trybie pauzy
                    # bez tego aplikacja nie reaguje na klawisze po wejściu w stan pauzy
                    self._handle_keys()

                    # Dodajemy krótkie opóźnienie, aby ograniczyć zużycie CPU w trybie pauzy
                    time.sleep(0.05)

                    # Przejście do następnej iteracji pętli
                    continue

                # Analizuj jasność otoczenia i aktualizuj kolory jesli włączona funkcja adaptacyjnego oświetlenia
                if self.adaptive_lighting and self.lighting_manager is not None:
                    try:
                        # Analiza jasności obrazu
                        brightness = self.lighting_manager.analyze_frame(frame)

                        # Aktualizacja kolorów na podstawie jasności
                        bg_color, figure_color = self.lighting_manager.update_colors(brightness)

                        # Ustawienie nowych kolorów dla renderera
                        self.stick_figure_renderer.set_colors(
                            bg_color=bg_color,
                            figure_color=figure_color
                        )

                        # Logowanie stanu co 100 klatek
                        if self.frame_count % 100 == 0:
                            colors = self.lighting_manager.get_current_colors()
                            self.logger.debug(
                                "AdaptiveLighting",
                                f"Jasność otoczenia: {brightness:.2f}, jasność tła: {colors['brightness_level']:.2f}",
                                log_type="LIGHTING"
                            )
                    except Exception as e:
                        self.logger.warning(
                            "Main",
                            f"Błąd w adaptacyjnym oświetleniu: {str(e)}",
                            log_type="LIGHTING"
                        )

                # 2. Detekcja pozy
                pose_detected, pose_data = self.pose_detector.detect_pose(frame)

                if not pose_detected:
                    # Jeśli nie wykryto pozy, pokazujemy oryginalny obraz lub pusty
                    if self.debug and self.show_preview:
                        self._show_preview(frame, None, None)

                    # Wysyłamy biały obraz (lub można użyć np. ostatniego poprawnego stick figure)
                    white_canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

                    if virtual_camera_working:
                        # Próbujemy wysłać klatkę tylko jeśli kamera działa
                        sent = self.virtual_camera.send_frame(white_canvas)
                        if not sent:
                            # Jeśli wysłanie się nie powiodło, aktualizujemy stan
                            virtual_camera_working = False
                            self.logger.warning(
                                "Main",
                                "Utracono połączenie z wirtualną kamerą",
                                log_type="VIRTUAL_CAM"
                            )

                    # Dodajemy obsługę klawiszy również tutaj,
                    # aby poprawić responsywność gdy nie wykryto pozy
                    self._handle_keys()
                    continue

                # 3. Analiza postawy
                posture_data = self.posture_analyzer.analyze_posture(
                    pose_data["landmarks"],
                    pose_data["frame_height"],
                    pose_data["frame_width"]
                )

                # 4. Renderowanie stick figure
                stick_figure = self.stick_figure_renderer.render(
                    pose_data["landmarks"],
                    posture_data["is_sitting"],
                    posture_data["confidence"]
                )

                # 5. Wysyłanie obrazu do wirtualnej kamery
                if virtual_camera_working:
                    sent = self.virtual_camera.send_frame(stick_figure)
                    if not sent:
                        # Jeśli wysłanie się nie powiodło, aktualizujemy stan i próbujemy zresetować
                        virtual_camera_working = False
                        self.logger.warning(
                            "Main",
                            "Utracono połączenie z wirtualną kamerą. Próba resetu...",
                            log_type="VIRTUAL_CAM"
                        )
                        # Próba resetu wirtualnej kamery
                        if self.virtual_camera.reset():
                            virtual_camera_working = self.virtual_camera.initialize()
                            if virtual_camera_working:
                                self.logger.info(
                                    "Main",
                                    "Przywrócono połączenie z wirtualną kamerą",
                                    log_type="VIRTUAL_CAM"
                                )

                # 6. Wyświetlanie podglądu
                if self.show_preview:
                    self._show_preview(frame, pose_data, stick_figure)

                # 7. Obsługa klawiszy
                self._handle_keys()

                # Pomiar wydajności
                self.performance.stop_timer()
                execution_time = self.performance.get_last_execution_time() * 1000  # ms

                # Logowanie wydajności co 100 klatek
                if self.frame_count % 100 == 0:
                    self.logger.performance_metrics(
                        self.current_fps,
                        execution_time,
                        "Main"
                    )

                # Limit FPS jeśli potrzeba
                frame_time = self.performance.get_last_execution_time()
                target_time = 1.0 / self.fps

                if frame_time < target_time:
                    time.sleep(target_time - frame_time)

        except KeyboardInterrupt:
            self.logger.info("Main", "Przerwanie przez użytkownika (Ctrl+C)", log_type="CONFIG")
        except Exception as e:
            self.logger.critical(
                "Main",
                f"Błąd w głównej pętli: {str(e)}",
                log_type="CONFIG",
                error={"error": str(e)}
            )
            raise
        finally:
            self._cleanup()

    def _show_preview(
        self,
        original_frame: np.ndarray,
        pose_data: Optional[Dict[str, Any]],
        stick_figure: Optional[np.ndarray]
    ) -> None:
        """
        Wyświetla podgląd obrazów w trybie debug.

        Args:
            original_frame (np.ndarray): Oryginalny obraz z kamery
            pose_data (Optional[Dict[str, Any]]): Dane wykrytej pozy lub None
            stick_figure (Optional[np.ndarray]): Wygenerowany stick figure lub None
        """
        if not self.show_preview:
            return

        try:
            # Tworzymy kopię oryginalnego obrazu
            preview = original_frame.copy()

            # Jeśli mamy dane pozy, wizualizujemy punkty charakterystyczne
            if pose_data is not None and "landmarks" in pose_data and pose_data["landmarks"] is not None:
                # Wizualizacja pozy
                preview = self.pose_detector.draw_pose_on_image(
                    preview,
                    pose_data["landmarks"],
                    draw_connections=True
                )

                # Dodatkowe informacje
                sitting_text = "Siedzi" if self.posture_analyzer.is_sitting else "Stoi"
                cv2.putText(
                    preview,
                    sitting_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            # Wyświetlamy FPS
            cv2.putText(
                preview,
                f"FPS: {self.current_fps:.1f}",
                (10, self.height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            # Status wirtualnej kamery
            vcam_status = "Virtual Camera: "
            if hasattr(self.virtual_camera, 'is_initialized') and self.virtual_camera.is_initialized:
                vcam_status += "ACTIVE"
                vcam_color = (0, 255, 0)  # Zielony
            elif hasattr(self.virtual_camera, 'initialization_failed') and self.virtual_camera.initialization_failed:
                vcam_status += "FAILED (press 'r' to reset)"
                vcam_color = (0, 0, 255)  # Czerwony
            else:
                vcam_status += "INACTIVE (press 'r' to initialize)"
                vcam_color = (0, 165, 255)  # Pomarańczowy

            cv2.putText(
                preview,
                vcam_status,
                (10, self.height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                vcam_color,
                1
            )

            # Wyświetlamy status adaptacyjnego oświetlenia
            if hasattr(self, 'lighting_manager') and self.lighting_manager is not None:
                adaptive_text = f"Adaptacyjne oswietlenie: {'ON' if self.adaptive_lighting else 'OFF'} (klawisz 'l')"
                cv2.putText(
                    preview,
                    adaptive_text,
                    (10, self.height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            # Status pauzy - POPRAWKA: Używamy ASCII zamiast polskich znaków
            if self.paused:
                # POPRAWKA: Usunięto polskie znaki, które powodowały problemy z wyświetlaniem
                pause_text = "PAUZA (nacisnij 'p' aby wznowic)"
                cv2.putText(
                    preview,
                    pause_text,
                    (self.width // 2 - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),  # Czerwony
                    2
                )

            # Klawisze sterujące
            controls_text = "Controls: q/ESC=exit, p=pause, r=reset vcam, t=test pattern, d=debug, l=lighting"
            cv2.putText(
                preview,
                controls_text,
                (10, self.height - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            # Wyświetlamy podgląd oryginalnego obrazu
            cv2.imshow("Podglad kamery", preview)

            # Jeśli mamy stick figure, wyświetlamy go również
            if stick_figure is not None:
                # Dodajemy informację o klawiszach do obrazu stick figure
                cv2.putText(
                    stick_figure,
                    vcam_status,
                    (10, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 100, 100) if self.virtual_camera.is_initialized else (50, 50, 200),
                    1
                )

                # POPRAWKA: Używamy ASCII zamiast polskich znaków w stick figure również
                if self.paused:
                    cv2.putText(
                        stick_figure,
                        "PAUZA",
                        (self.width // 2 - 50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (100, 100, 150),  # Przyciemniony czerwony dla stick figure
                        2
                    )

                cv2.imshow("Stick Figure", stick_figure)

        except Exception as e:
            self.logger.error(
                "Main",
                f"Błąd podczas wyświetlania podglądu: {str(e)}",
                log_type="CONFIG"
            )

    def _handle_keys(self) -> None:
        """
        Obsługuje wciśnięcia klawiszy do sterowania aplikacją.
        """
        # Zwiększono czas oczekiwania w trybie pauzy dla lepszej responsywności,
        # W trybie normalnym zostawiamy 1ms, w trybie pauzy zwiększamy do 10ms
        wait_time = 10 if self.paused else 1
        key = cv2.waitKey(wait_time) & 0xFF

        if key == 27 or key == ord('q'):  # ESC lub q - wyjście
            self.running = False
            self.logger.info("Main", "Wyjście z aplikacji (klawisz q/ESC)", log_type="CONFIG")

        elif key == ord('p'):  # p - pauza/wznowienie
            self.paused = not self.paused
            status = "Wstrzymano" if self.paused else "Wznowiono"
            self.logger.info(
                "Main",
                f"{status} przetwarzanie (klawisz p)",
                log_type="CONFIG"
            )

            # Jeśli wznawiamy po pauzie, wyczyść bufor klawiatury OpenCV
            # aby nie przetwarzać klawiszy naciśniętych podczas pauzy
            if not self.paused:
                cv2.waitKey(1)

        elif key == ord('d'):  # d - przełączenie trybu debug
            self.debug = not self.debug
            self.logger.info(
                "Main",
                f"Tryb debug {'włączony' if self.debug else 'wyłączony'} (klawisz d)",
                log_type="CONFIG"
            )

            # Aktualizacja poziomu logowania
            for handler in self.logger.logger.handlers:
                if isinstance(handler, type(self.logger.logger.handlers[0])):  # Console handler
                    handler.setLevel(logging.DEBUG if self.debug else logging.INFO)

        elif key == ord('f'):  # f - przełączenie odbicia poziomego
            self.flip_camera = not self.flip_camera
            self.logger.info(
                "Main",
                f"Odbicie poziome {'włączone' if self.flip_camera else 'wyłączone'} (klawisz f)",
                log_type="CONFIG"
            )

        elif key == ord('t'):  # t - wysłanie wzoru testowego
            success = self.virtual_camera.send_test_pattern()
            self.logger.info(
                "Main",
                f"Wysłano wzór testowy (klawisz t) - {'sukces' if success else 'niepowodzenie'}",
                log_type="VIRTUAL_CAM"
            )

        elif key == ord('r'):  # r - reset wirtualnej kamery
            self.logger.info("Main", "Resetowanie wirtualnej kamery (klawisz r)", log_type="VIRTUAL_CAM")
            if self.virtual_camera.reset() and self.virtual_camera.initialize():
                self.logger.info("Main", "Reset wirtualnej kamery zakończony powodzeniem", log_type="VIRTUAL_CAM")
                self.virtual_camera.send_test_pattern()
            else:
                self.logger.warning("Main", "Reset wirtualnej kamery nie powiódł się", log_type="VIRTUAL_CAM")

        elif key == ord('l'):  # l - przełączenie adaptacyjnego oświetlenia
            # Sprawdzenie czy lighting_manager jest zainicjowany
            if not hasattr(self, 'lighting_manager') or self.lighting_manager is None:
                # Inicjalizuj manager oświetlenia podczas pierwszego włączenia
                self.lighting_manager = AdaptiveLightingManager(
                    adaptation_speed=0.01,  # Domyślna wartość
                    smoothing_window=int(self.fps),  # 1 sekunda jako okno wygładzania
                    sampling_interval=3,  # Analizuj co 3 klatkę dla oszczędności CPU
                    logger=self.logger
                )
                self.logger.info(
                    "Main",
                    "Utworzono manager adaptacyjnego oświetlenia",
                    log_type="LIGHTING"
                )

            # Przełącz stan
            self.adaptive_lighting = not self.adaptive_lighting
            self.logger.info(
                "Main",
                f"Adaptacyjne oświetlenie {'włączone' if self.adaptive_lighting else 'wyłączone'} (klawisz l)",
                log_type="LIGHTING"
            )

            # Jeśli wyłączono adaptacyjne oświetlenie, przywróć domyślne kolory
            if not self.adaptive_lighting:
                self.stick_figure_renderer.set_colors(
                    bg_color=(255, 255, 255),  # Białe tło
                    figure_color=(0, 0, 0)  # Czarny kontur
                )

    def _cleanup(self) -> None:
        """
        Zwalnia zasoby i zamyka połączenia przed zakończeniem aplikacji.
        """
        self.logger.info("Main", "Zamykanie zasobów...", log_type="CONFIG")

        try:
            # Zamykanie podglądu
            cv2.destroyAllWindows()

            # Zamykanie kamery
            if hasattr(self, 'camera'):
                self.camera.close()

            # Zamykanie wirtualnej kamery
            if hasattr(self, 'virtual_camera'):
                self.virtual_camera.close()

            # Zamykanie detektora pozy
            if hasattr(self, 'pose_detector'):
                self.pose_detector.close()

            # Resetowanie managera adaptacyjnego oświetlenia
            if hasattr(self, 'lighting_manager') and self.lighting_manager is not None:
                self.lighting_manager.reset()

            self.logger.info("Main", "Wszystkie zasoby zamknięte pomyślnie", log_type="CONFIG")

        except Exception as e:
            self.logger.error(
                "Main",
                f"Błąd podczas zwalniania zasobów: {str(e)}",
                log_type="CONFIG"
            )


def parse_arguments():
    """
    Parsuje argumenty linii poleceń.

    Returns:
        argparse.Namespace: Sparsowane argumenty
    """
    parser = argparse.ArgumentParser(description="Stick Figure Webcam - zamień siebie w animowaną postać stick figure")

    parser.add_argument("-c", "--camera", type=int, default=0,
                        help="Numer identyfikacyjny kamery (domyślnie 0)")
    parser.add_argument("-w", "--width", type=int, default=640,
                        help="Szerokość obrazu (domyślnie 640)")
    parser.add_argument("-H", "--height", type=int, default=480,
                        help="Wysokość obrazu (domyślnie 480)")
    parser.add_argument("-f", "--fps", type=int, default=30,
                        help="Docelowa liczba klatek na sekundę (domyślnie 30)")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Włącza tryb debugowania")
    parser.add_argument("-l", "--log-level", type=str, default="INFO",
                        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Poziom logowania (domyślnie INFO)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Ścieżka do pliku logów (domyślnie logs/stick_figure_webcam_YYYYMMDD_HHMMSS.log)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Wyłącza podgląd obrazu")
    parser.add_argument("--no-flip", action="store_true",
                        help="Wyłącza automatyczne odbicie poziome obrazu")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Pomija sprawdzanie wymagań systemowych")
    # Dodane argumenty dla adaptacyjnego oświetlenia
    parser.add_argument("--adaptive-lighting", action="store_true",
                        help="Włącza adaptacyjne dostosowywanie kolorów do oświetlenia otoczenia")
    parser.add_argument("--adaptation-speed", type=float, default=0.01,
                        help="Szybkość adaptacji kolorów (0.001-0.1, domyślnie 0.01)")

    return parser.parse_args()


def main():
    """
    Główna funkcja uruchamiająca aplikację.
    """
    # Ignorowanie ostrzeżeń NumPy w trybie produkcyjnym
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # Parsowanie argumentów linii poleceń
    args = parse_arguments()

    # Tworzymy katalog logs bezwarunkowo, jeśli nie istnieje
    try:
        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)
            print(f"Utworzono katalog logów: {os.path.abspath('logs')}")
    except Exception as e:
        print(f"UWAGA: Nie można utworzyć katalogu logów: {e}")

    # Konfiguruję ścieżkę pliku logów
    log_file = args.log_file
    if log_file is None:
        # Generujemy domyślną nazwę pliku logów zawierającą datę i czas
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join("logs", f"stick_figure_webcam_{timestamp}.log")
        print(f"Używam domyślnego pliku logów: {log_file}")

    log_level = args.log_level
    debug = args.debug

    # Inicjalizacja loggera z explicit ustawionym plikiem logów
    try:
        logger = CustomLogger(
            log_file=log_file,  # Zawsze przekazujemy ścieżkę pliku
            console_level="DEBUG" if debug else log_level,
            file_level="DEBUG",  # Zawsze zapisujemy szczegółowe logi do pliku
            verbose=debug  # Dodatkowe szczegóły dla trybu debug
        )
        logger.info("Main", f"Inicjalizacja logowania do pliku: {log_file}", log_type="CONFIG")
    except Exception as e:
        print(f"BŁĄD: Nie można zainicjalizować loggera: {e}")
        # Fallback do podstawowego loggera, jeśli CustomLogger zawiedzie
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("fallback")
        logger.error(f"Nie można zainicjalizować CustomLogger: {e}")

    logger.info("Main", "Uruchamianie Stick Figure Webcam", log_type="CONFIG")

    # Sprawdzenie wymagań systemowych
    if not args.skip_checks:
        logger.info("Main", "Sprawdzanie wymagań systemowych...", log_type="CONFIG")
        all_requirements_met, system_check_results = check_system_requirements(logger)

        if not all_requirements_met:
            logger.warning(
                "Main",
                "Nie wszystkie wymagania systemowe są spełnione. Wyświetlanie dialogu konfiguracyjnego.",
                log_type="CONFIG"
            )

            # Zmienne do obsługi wywołań zwrotnych dialogu
            app_started = [False]  # Lista jako hack, żeby móc modyfikować zmienną w callbacku

            def start_app():
                app_started[0] = True

            # Wyświetlenie dialogu konfiguracyjnego
            show_setup_dialog(
                system_check_results,
                on_continue=start_app,
                on_exit=lambda: sys.exit(0),
                logger=logger
            )

            # Jeśli użytkownik nie zdecydował się kontynuować, kończymy
            if not app_started[0]:
                logger.info("Main", "Kończenie działania aplikacji", log_type="CONFIG")
                return

    # Tworzenie i uruchamianie aplikacji
    try:
        app = StickFigureWebcam(
            camera_id=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            debug=debug,
            log_level=log_level,
            log_file=log_file,
            show_preview=not args.no_preview,
            flip_camera=not args.no_flip,
            adaptive_lighting=args.adaptive_lighting,
            adaptation_speed=args.adaptation_speed,
            logger=logger  # Przekazujemy już utworzony logger
        )

        app.run()
    except Exception as e:
        logger.critical("Main", f"Krytyczny błąd aplikacji: {str(e)}", log_type="CONFIG")
        print(f"\nKrytyczny błąd aplikacji: {str(e)}")
        print("Sprawdź logi, aby uzyskać więcej informacji.")
        sys.exit(1)


if __name__ == "__main__":
    main()
