#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/main.py

"""
Główny skrypt dla Reaktywnego Stick Figure Webcam dla Discorda.
Przechwytuje obraz z kamery, wykrywa twarz i ręce, a następnie generuje animowaną postać patyczaka.
Wersja z płynną animacją rąk i loggerem z modułu utils.
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Dict, Any, Optional, List

import cv2
import mediapipe as mp

from src.drawing.stick_figure_renderer import StickFigureRenderer
# Importy z własnych modułów
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class StickFigureWebcam:
    """
    Główna klasa aplikacji, która łączy wszystkie komponenty i zarządza przepływem danych.
    Wykrywa twarz i ręce, a następnie renderuje animowaną postać stick figure.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        debug: bool = False,
        show_preview: bool = True,
        flip_camera: bool = True,
        use_virtual_camera: bool = True,
        show_face_mesh: bool = True,
        log_file: Optional[str] = None,
        performance_log_interval: int = 30  # Co ile sekund logować wydajność
    ):
        """
        Inicjalizacja aplikacji Stick Figure Webcam.

        Args:
            camera_id (int): Identyfikator kamery (0 dla domyślnej)
            width (int): Szerokość obrazu
            height (int): Wysokość obrazu
            fps (int): Docelowa liczba klatek na sekundę
            debug (bool): Czy włączyć tryb debugowania
            show_preview (bool): Czy pokazywać podgląd obrazu
            flip_camera (bool): Czy odbijać obraz z kamery w poziomie
            use_virtual_camera (bool): Czy używać wirtualnej kamery
            show_face_mesh (bool): Czy pokazywać siatkę twarzy na podglądzie
            log_file (Optional[str]): Ścieżka do pliku logów, jeśli None to logi tylko na konsolę
            performance_log_interval (int): Co ile sekund logować statystyki wydajności
        """
        # Inicjalizacja loggera
        self.logger = CustomLogger(
            log_file=log_file,
            console_level="INFO" if not debug else "DEBUG",
            file_level="DEBUG",
            verbose=debug
        )

        # Rejestracja obsługi sygnałów dla bezpiecznego zamykania
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Konfiguracja
        self.width = width
        self.height = height
        self.fps = fps
        self.debug = debug
        self.show_preview = show_preview
        self.flip_camera = flip_camera
        self.use_virtual_camera = use_virtual_camera
        self.show_face_mesh = show_face_mesh
        self.performance_log_interval = performance_log_interval

        # Flagi stanu
        self.running = False
        self.paused = False
        self.last_performance_log_time = 0  # Czas ostatniego logowania wydajności

        self.logger.info(
            "Main",
            f"Inicjalizacja Stick Figure Webcam ({width}x{height} @ {fps}FPS)",
            log_type="CONFIG"
        )

        # Monitor wydajności
        self.performance = PerformanceMonitor("MainLoop")

        # Inicjalizacja komponentów
        try:
            # Inicjalizacja kamery
            self.logger.info("Main", "Inicjalizacja kamery...", log_type="CAMERA")
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)

            if not self.camera.isOpened():
                raise RuntimeError("Nie można otworzyć kamery")

            # Logowanie statusu kamery
            camera_info = {
                "name": f"Camera {camera_id}",
                "resolution": (
                    int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                "fps": self.camera.get(cv2.CAP_PROP_FPS)
            }
            self.logger.camera_status(True, camera_info)

            # Inicjalizacja wirtualnej kamery (jeśli wymagana)
            self.virtual_camera_ready = False
            if use_virtual_camera:
                try:
                    import pyvirtualcam
                    self.logger.info("Main", "Inicjalizacja wirtualnej kamery...", log_type="VIRTUAL_CAM")
                    self.virtual_camera = pyvirtualcam.Camera(
                        width=width,
                        height=height,
                        fps=fps
                    )
                    self.virtual_camera_ready = True
                    self.logger.info(
                        "Main",
                        f"Wirtualna kamera gotowa: {self.virtual_camera.device}",
                        log_type="VIRTUAL_CAM"
                    )

                    # Logowanie szczegółów wirtualnej kamery
                    virtual_camera_info = {
                        "device": self.virtual_camera.device,
                        "resolution": (width, height),
                        "fps": fps,
                        "backend": self.virtual_camera.backend
                    }
                    self.logger.virtual_camera_status(True, virtual_camera_info)
                except Exception as e:
                    self.logger.error(
                        "Main",
                        f"Nie można zainicjalizować wirtualnej kamery: {e}",
                        log_type="VIRTUAL_CAM",
                        error={"error": str(e)}
                    )
                    self.logger.warning(
                        "Main",
                        "Aplikacja będzie działać bez wirtualnej kamery.",
                        log_type="VIRTUAL_CAM"
                    )

            # Inicjalizacja detektora twarzy MediaPipe
            self.logger.info("Main", "Inicjalizacja detektora twarzy...", log_type="FACE")
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Inicjalizacja detektora rąk MediaPipe - dodajemy to, aby faktycznie wykrywać ręce
            self.logger.info("Main", "Inicjalizacja detektora rąk...", log_type="HANDS")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Inicjalizacja renderera stick figure
            self.logger.info("Main", "Inicjalizacja renderera stick figure...", log_type="DRAWING")
            self.renderer = StickFigureRenderer(
                canvas_width=width,
                canvas_height=height,
                line_thickness=3,
                head_radius_factor=0.075,
                bg_color=(255, 255, 255),  # Białe tło
                figure_color=(0, 0, 0),  # Czarny patyczak
                smooth_factor=0.3,
                logger=self.logger
            )

            # Liczniki i statystyki
            self.frame_count = 0
            self.fps_counter = 0
            self.fps_timer = time.time()
            self.current_fps = 0.0

            # Ostatnie wykryte wartości ekspresji do wyświetlania na ekranie
            self.last_expression_values = {
                "mouth_open": 0.0,
                "smile": 0.5
            }

            self.logger.info("Main", "Stick Figure Webcam zainicjalizowany pomyślnie", log_type="CONFIG")

        except Exception as e:
            self.logger.critical(
                "Main",
                f"Błąd podczas inicjalizacji: {str(e)}",
                log_type="CONFIG",
                error={"error": str(e)}
            )
            raise

    def run(self):
        """
        Uruchamia główną pętlę aplikacji.
        """
        self.running = True
        self.logger.info("Main", "Uruchamianie głównej pętli aplikacji", log_type="CONFIG")
        self.logger.info(
            "Main",
            "Naciśnij 'q' aby zakończyć, 'p' aby wstrzymać/wznowić, 's' aby zmienić nastrój",
            log_type="CONFIG"
        )
        self.logger.info(
            "Main",
            "Naciśnij 'm' aby włączyć/wyłączyć wizualizację siatki twarzy, 'd' aby włączyć/wyłączyć tryb debugowania",
            log_type="CONFIG"
        )

        try:
            # Główna pętla aplikacji
            while self.running:
                self.performance.start_timer()

                # Obliczanie FPS
                current_time = time.time()
                if current_time - self.fps_timer >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_timer)
                    self.fps_timer = current_time
                    self.fps_counter = 0

                    # Logowanie FPS co określony interwał (zamiast co sekundę)
                    if current_time - self.last_performance_log_time >= self.performance_log_interval:
                        self.logger.performance_metrics(
                            self.current_fps,
                            1000.0 / max(1, self.current_fps),  # ms per frame
                            "MainLoop"
                        )
                        self.last_performance_log_time = current_time

                # W trybie pauzy tylko sprawdzamy klawisze
                if self.paused:
                    self._handle_keys()
                    time.sleep(0.05)  # Zmniejszamy obciążenie CPU
                    continue

                # 1. Pobieranie klatki z kamery
                ret, frame = self.camera.read()

                if not ret:
                    self.logger.warning("Main", "Nie udało się odczytać klatki z kamery", log_type="CAMERA")
                    time.sleep(0.1)
                    continue

                # Odbicie obrazu w poziomie jeśli potrzeba
                if self.flip_camera:
                    frame = cv2.flip(frame, 1)  # 1 = odbicie poziome

                # 2. Przetwarzanie obrazu
                # Konwersja do RGB (wymagane przez MediaPipe)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 3. Detekcja twarzy
                face_results = self.face_mesh.process(rgb_frame)

                # 4. Detekcja rąk
                hands_results = self.hands.process(rgb_frame)

                # Kopiowanie obrazu do wyświetlenia z zaznaczeniami
                debug_frame = frame.copy()

                # Przetworzenie wyników detekcji twarzy i rąk
                face_data = None

                if face_results.multi_face_landmarks:
                    # Przetwarzanie punktów charakterystycznych twarzy
                    face_data = self._process_face_landmarks(face_results.multi_face_landmarks[0])

                    # Jeśli mamy wykryte punkty twarzy i opcja wizualizacji jest włączona,
                    # rysujemy siatkę na obrazie debugowania
                    if self.show_face_mesh:
                        for face_landmarks in face_results.multi_face_landmarks:
                            # Rysujemy siatkę twarzy na obrazie podglądu
                            self.mp_drawing.draw_landmarks(
                                image=debug_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )

                            # Dodatkowo rysujemy kontury oczu i ust dla lepszej wizualizacji
                            self.mp_drawing.draw_landmarks(
                                image=debug_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )

                    # Zapisujemy ostatnie wykryte wartości ekspresji
                    if "expressions" in face_data:
                        self.last_expression_values = face_data["expressions"]

                # Dodajemy informacje o rękach do danych twarzy
                if hands_results.multi_hand_landmarks:
                    if face_data is None:
                        face_data = {
                            "has_face": False,
                            "landmarks": [],
                            "expressions": self.last_expression_values
                        }

                    # Dodajemy punkty rąk do danych
                    self._process_hand_landmarks(hands_results.multi_hand_landmarks, face_data)

                    # Wyświetlamy wykryte ręce w trybie debug
                    if self.show_preview:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                debug_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                # 5. Renderowanie stick figure
                output_image = self.renderer.render(face_data)

                # 6. Wysyłanie obrazu do wirtualnej kamery
                if self.virtual_camera_ready:
                    # Konwersja BGR -> RGB dla pyvirtualcam
                    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    self.virtual_camera.send(output_rgb)

                # 7. Wyświetlanie podglądu
                if self.show_preview:
                    self._show_preview(debug_frame, output_image)

                # 8. Obsługa klawiszy
                self._handle_keys()

                # Aktualizacja liczników
                self.frame_count += 1
                self.fps_counter += 1

                # Mierzenie wydajności
                self.performance.stop_timer()
                frame_time = self.performance.get_last_execution_time()

                # Limit FPS - sleep tylko gdy potrzeba
                elapsed_time = frame_time
                target_time = 1.0 / self.fps

                if elapsed_time < target_time:
                    time.sleep(target_time - elapsed_time)

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

    def _process_face_landmarks(self, face_landmarks) -> Dict[str, Any]:
        """
        Przetwarza punkty charakterystyczne twarzy z MediaPipe na format używany przez nasz renderer.

        Args:
            face_landmarks: Punkty charakterystyczne twarzy z MediaPipe

        Returns:
            dict: Słownik z danymi twarzy
        """
        try:
            # Konwersja punktów MediaPipe do formatu używanego przez nasz renderer
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility
                if hasattr(landmark, 'visibility') else 1.0))

            # Analizujemy kluczowe punkty twarzy, aby określić wyrazy mimiczne
            mouth_open = 0.0
            smile = 0.5  # Neutralny uśmiech

            try:
                # Otwartość ust - używamy punktów górnej i dolnej wargi
                upper_lip = face_landmarks.landmark[13]  # Górna warga
                lower_lip = face_landmarks.landmark[14]  # Dolna warga

                # Odległość między wargami wskazuje na otwartość ust
                # Normalizacja do zakresu 0-1
                mouth_height = abs(lower_lip.y - upper_lip.y)

                # Zwiększamy czułość (mnożymy przez 20 zamiast 10)
                mouth_open = min(1.0, max(0.0, mouth_height * 20))

                # Uśmiech - używamy punktów kącików ust i ich pozycji względem środka ust
                left_corner = face_landmarks.landmark[61]  # Lewy kącik ust
                right_corner = face_landmarks.landmark[291]  # Prawy kącik ust
                center_mouth = face_landmarks.landmark[13]  # Górna warga jako punkt odniesienia

                # Uśmiech określamy na podstawie pozycji kącików ust względem środka
                # Jeśli kąciki są wyżej niż środek, to uśmiech
                # Jeśli niżej, to smutek
                corner_height_avg = (left_corner.y + right_corner.y) / 2

                # Zwiększamy czułość na zmiany
                height_diff = center_mouth.y - corner_height_avg

                # Debug - wypisujemy wartości różnicy wysokości
                if self.debug and self.frame_count % 30 == 0:
                    self.logger.debug(
                        "Main",
                        f"Różnica wysokości kącików ust: {height_diff:.6f}",
                        log_type="FACE"
                    )

                if height_diff > 0.005:  # Próg dla uśmiechu
                    # Kąciki ust są wyżej - uśmiech
                    # Siła uśmiechu zależy od tego, jak wysoko są kąciki
                    smile_strength = height_diff * 10  # Zwiększona czułość
                    smile = 0.5 + min(0.5, smile_strength)  # Zakres 0.5-1.0
                elif height_diff < -0.005:  # Próg dla smutku
                    # Kąciki ust są niżej - smutek
                    # Siła smutku zależy od tego, jak nisko są kąciki
                    sad_strength = -height_diff * 10  # Zwiększona czułość
                    smile = 0.5 - min(0.5, sad_strength)  # Zakres 0.0-0.5
                else:
                    # Kąciki mniej więcej na poziomie środka - neutralny wyraz
                    smile = 0.5

                # Dodatkowy debug wartości
                if self.debug and self.frame_count % 30 == 0:
                    self.logger.debug(
                        "Main",
                        f"Wartość smile: {smile:.2f}, mouth_open: {mouth_open:.2f}",
                        log_type="FACE"
                    )

            except Exception as e:
                self.logger.error(
                    "Main",
                    f"Błąd podczas analizy mimiki twarzy: {str(e)}",
                    log_type="FACE",
                    error={"error": str(e)}
                )

            # Tworzymy słownik z danymi twarzy
            return {
                "has_face": True,
                "landmarks": landmarks,
                "expressions": {
                    "mouth_open": mouth_open,
                    "smile": smile,
                    "left_eye_open": 1.0,
                    "right_eye_open": 1.0,
                }
            }
        except Exception as e:
            self.logger.error(
                "Main",
                f"Błąd podczas przetwarzania punktów twarzy: {str(e)}",
                log_type="FACE",
                error={"error": str(e)}
            )
            return {"has_face": False}

    def _process_hand_landmarks(self, multi_hand_landmarks: List, face_data: Dict[str, Any]) -> None:
        """
        Przetwarza punkty charakterystyczne rąk z MediaPipe i dodaje je do danych twarzy.

        Args:
            multi_hand_landmarks: Lista punktów charakterystycznych rąk z MediaPipe
            face_data: Słownik z danymi twarzy, do którego dodajemy informacje o rękach
        """
        try:
            # Indeksy kluczowych punktów rąk w MediaPipe Hands
            WRIST = 0  # Nadgarstek
            THUMB_CMC = 1  # Podstawa kciuka
            INDEX_MCP = 5  # Podstawa palca wskazującego
            MIDDLE_MCP = 9  # Podstawa palca środkowego
            PINKY_MCP = 17  # Podstawa małego palca

            # Dodajemy informacje o rękach do danych
            if not "hands_data" in face_data:
                face_data["hands_data"] = {
                    "left_hand": None,
                    "right_hand": None
                }

            for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
                # Próba określenia, czy to lewa czy prawa ręka
                # Używamy prostej heurystyki - jeśli kciuk jest po lewej stronie nadgarstka, to lewa ręka
                wrist_x = hand_landmarks.landmark[WRIST].x
                thumb_x = hand_landmarks.landmark[THUMB_CMC].x
                index_x = hand_landmarks.landmark[INDEX_MCP].x
                pinky_x = hand_landmarks.landmark[PINKY_MCP].x

                # W kamerze z odbiciem poziomym (flipped) trzeba odwrócić logikę
                is_left_hand = (thumb_x < wrist_x) if not self.flip_camera else (thumb_x > wrist_x)

                # Przekształcamy punkty MediaPipe na nasz format
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.append((landmark.x, landmark.y, landmark.z,
                                        landmark.visibility if hasattr(landmark, 'visibility') else 1.0))

                # Obliczenie pozycji łokcia na podstawie nadgarstka i środka ciała
                # Jest to szacunek, ponieważ MediaPipe Hands nie wykrywa łokcia
                # Używamy pozycji nadgarstka i estymujemy łokieć w kierunku ciała
                wrist_pos = hand_points[WRIST]
                center_x = 0.5  # Środek ekranu w poziomie
                # Obliczamy wektor od nadgarstka do środka ciała
                vector_x = center_x - wrist_pos[0]
                vector_y = 0.2 - wrist_pos[1]  # Zakładamy, że środek ciała jest wyżej niż nadgarstek
                # Normalizujemy wektor
                vector_len = (vector_x ** 2 + vector_y ** 2) ** 0.5
                if vector_len > 0:
                    vector_x /= vector_len
                    vector_y /= vector_len
                # Tworzymy punkt łokcia pomiędzy nadgarstkiem a ciałem
                elbow_x = wrist_pos[0] + vector_x * 0.15
                elbow_y = wrist_pos[1] + vector_y * 0.15
                elbow_pos = (elbow_x, elbow_y, 0.0, 1.0)

                # Dodajemy kluczowe punkty do danych
                hand_data = {
                    "wrist": wrist_pos,
                    "elbow": elbow_pos,
                    "is_left": is_left_hand
                }

                # Dodajemy do odpowiedniej ręki
                if is_left_hand:
                    face_data["hands_data"]["left_hand"] = hand_data
                    if self.debug and self.frame_count % 50 == 0:
                        self.logger.debug(
                            "Main",
                            f"Wykryto LEWĄ rękę: nadgarstek=({wrist_pos[0]:.2f}, {wrist_pos[1]:.2f})",
                            log_type="HANDS"
                        )
                else:
                    face_data["hands_data"]["right_hand"] = hand_data
                    if self.debug and self.frame_count % 50 == 0:
                        self.logger.debug(
                            "Main",
                            f"Wykryto PRAWĄ rękę: nadgarstek=({wrist_pos[0]:.2f}, {wrist_pos[1]:.2f})",
                            log_type="HANDS"
                        )

        except Exception as e:
            self.logger.error(
                "Main",
                f"Błąd podczas przetwarzania punktów rąk: {str(e)}",
                log_type="HANDS",
                error={"error": str(e)}
            )

    def _show_preview(self, original_frame, stick_figure):
        """
        Wyświetla podgląd obrazów z dodatkowymi informacjami debugującymi.

        Args:
            original_frame: Oryginalny obraz z kamery z oznaczeniami
            stick_figure: Wygenerowany stick figure
        """
        try:
            # Wyświetlamy podgląd z kamery z oznaczeniami

            # Dodajemy informacje o FPS
            cv2.putText(
                original_frame,
                f"FPS: {self.current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Dodajemy informacje o rozpoznanych ekspresji
            if self.debug:
                smile_value = self.last_expression_values.get("smile", 0.5)
                mouth_open = self.last_expression_values.get("mouth_open", 0.0)

                # Określamy stan nastroju na podstawie wartości smile
                mood = "neutral"
                if smile_value > 0.6:
                    mood = "happy"
                elif smile_value < 0.4:
                    mood = "sad"

                # Dodajemy informacje o ekspresji
                expression_text = f"Smile: {smile_value:.2f} ({mood})"
                cv2.putText(
                    original_frame,
                    expression_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                mouth_text = f"Mouth open: {mouth_open:.2f}"
                cv2.putText(
                    original_frame,
                    mouth_text,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Dodajemy informację o stanie wizualizacji siatki twarzy
            mesh_text = "Face mesh: ON (klawisz 'm')" if self.show_face_mesh else "Face mesh: OFF (klawisz 'm')"
            cv2.putText(
                original_frame,
                mesh_text,
                (10, self.height - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Dodajemy informację o trybie debugowania
            debug_text = "Debug: ON (klawisz 'd')" if self.debug else "Debug: OFF (klawisz 'd')"
            cv2.putText(
                original_frame,
                debug_text,
                (10, self.height - 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Tryb pauzy
            if self.paused:
                cv2.putText(
                    original_frame,
                    "PAUZA (klawisz 'p')",
                    (self.width // 2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Podgląd", original_frame)

            # Jeśli mamy stick figure, wyświetlamy go również
            if stick_figure is not None:
                # Dodajemy informacje na ekranie stick figure
                cv2.putText(
                    stick_figure,
                    f"FPS: {self.current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 100, 100),
                    1
                )

                if self.paused:
                    cv2.putText(
                        stick_figure,
                        "PAUZA",
                        (self.width // 2 - 50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (100, 100, 100),
                        2
                    )

                cv2.imshow("Stick Figure", stick_figure)

        except Exception as e:
            self.logger.error(
                "Main",
                f"Błąd podczas wyświetlania podglądu: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)}
            )

    def _handle_keys(self):
        """
        Obsługuje wciśnięcia klawiszy.
        """
        wait_time = 10 if self.paused else 1
        key = cv2.waitKey(wait_time) & 0xFF

        if key == 27 or key == ord('q'):  # ESC lub q - wyjście
            self.running = False
            self.logger.info("Main", "Wyjście z aplikacji", log_type="CONFIG")

        elif key == ord('p'):  # p - pauza/wznowienie
            self.paused = not self.paused
            status = "Wstrzymano" if self.paused else "Wznowiono"
            self.logger.info("Main", f"{status} przetwarzanie", log_type="CONFIG")

        elif key == ord('f'):  # f - przełączenie odbicia poziomego
            self.flip_camera = not self.flip_camera
            self.logger.info(
                "Main",
                f"Odbicie poziome: {'włączone' if self.flip_camera else 'wyłączone'}",
                log_type="CONFIG"
            )

        elif key == ord('m'):  # m - włączenie/wyłączenie wizualizacji siatki twarzy
            self.show_face_mesh = not self.show_face_mesh
            self.logger.info(
                "Main",
                f"Wizualizacja siatki twarzy: {'włączona' if self.show_face_mesh else 'wyłączona'}",
                log_type="CONFIG"
            )

        elif key == ord('d'):  # d - włączenie/wyłączenie trybu debugowania
            self.debug = not self.debug
            self.logger.info(
                "Main",
                f"Tryb debugowania: {'włączony' if self.debug else 'wyłączony'}",
                log_type="CONFIG"
            )

            # Aktualizacja poziomu logowania
            if hasattr(self.logger, 'logger'):
                for handler in self.logger.logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        handler.setLevel(logging.DEBUG if self.debug else logging.INFO)

        elif key == ord('s'):  # s - zmiana nastroju
            moods = ["happy", "neutral", "sad", "surprised", "wink"]
            current_mood = self.renderer.mood

            # Znajdź obecny nastrój na liście i przejdź do następnego
            try:
                idx = moods.index(current_mood)
                new_idx = (idx + 1) % len(moods)
                new_mood = moods[new_idx]
            except ValueError:
                new_mood = "neutral"  # Domyślny nastrój jeśli obecny nie jest na liście

            self.renderer.set_mood(new_mood)
            self.logger.info("Main", f"Zmieniono nastrój na: {new_mood}", log_type="DRAWING")

    def _cleanup(self):
        """
        Zwalnia zasoby przed zakończeniem.
        """
        self.logger.info("Main", "Zamykanie zasobów...", log_type="CONFIG")

        try:
            # Zamykanie kamery
            if hasattr(self, 'camera'):
                self.camera.release()
                self.logger.debug("Main", "Zamknięto kamerę", log_type="CAMERA")

            # Zamykanie wirtualnej kamery
            if hasattr(self, 'virtual_camera_ready') and self.virtual_camera_ready:
                self.virtual_camera.close()
                self.logger.debug("Main", "Zamknięto wirtualną kamerę", log_type="VIRTUAL_CAM")

            # Zamykanie MediaPipe Face Mesh
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
                self.logger.debug("Main", "Zamknięto detektor twarzy", log_type="FACE")

            # Zamykanie MediaPipe Hands
            if hasattr(self, 'hands'):
                self.hands.close()
                self.logger.debug("Main", "Zamknięto detektor rąk", log_type="HANDS")

            # Zamykanie okien OpenCV
            cv2.destroyAllWindows()

            self.logger.info("Main", "Wszystkie zasoby zamknięte pomyślnie", log_type="CONFIG")

        except Exception as e:
            self.logger.error(
                "Main",
                f"Błąd podczas zwalniania zasobów: {str(e)}",
                log_type="CONFIG",
                error={"error": str(e)}
            )

    def signal_handler(self, sig, frame):
        """
        Obsługuje sygnały zewnętrzne (np. SIGINT, SIGTERM).

        Args:
            sig: Numer sygnału
            frame: Ramka stosu
        """
        self.logger.info("Main", f"Otrzymano sygnał {sig}, zamykanie...", log_type="CONFIG")
        self.running = False


def parse_arguments():
    """
    Parsuje argumenty linii poleceń.

    Returns:
        argparse.Namespace: Sparsowane argumenty
    """
    parser = argparse.ArgumentParser(description="Stick Figure Webcam - zamień siebie w animowaną postać patyczaka")

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
    parser.add_argument("--no-preview", action="store_true",
                        help="Wyłącza podgląd obrazu")
    parser.add_argument("--no-flip", action="store_true",
                        help="Wyłącza automatyczne odbicie poziome obrazu")
    parser.add_argument("--no-virtual-camera", action="store_true",
                        help="Wyłącza wirtualną kamerę")
    parser.add_argument("--no-face-mesh", action="store_true",
                        help="Wyłącza wizualizację siatki twarzy")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Ścieżka do pliku logów (domyślnie tylko logi na konsolę)")
    parser.add_argument("--perf-log-interval", type=int, default=30,
                        help="Co ile sekund logować statystyki wydajności (domyślnie 30)")

    return parser.parse_args()


def main():
    """
    Główna funkcja programu.
    """
    # Ignorowanie ostrzeżeń
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # Parsowanie argumentów
    args = parse_arguments()

    # Sprawdzenie czy istnieje katalog logs
    if args.log_file is not None:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # Tworzenie i uruchamianie aplikacji
    try:
        app = StickFigureWebcam(
            camera_id=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            debug=args.debug,
            show_preview=not args.no_preview,
            flip_camera=not args.no_flip,
            use_virtual_camera=not args.no_virtual_camera,
            show_face_mesh=not args.no_face_mesh,
            log_file=args.log_file,
            performance_log_interval=args.perf_log_interval
        )

        app.run()
    except Exception as e:
        print(f"Krytyczny błąd aplikacji: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
