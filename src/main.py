#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py - Wersja z rozszerzonym debugowaniem

"""
Główny skrypt dla Reaktywnego Stick Figure Webcam dla Discorda.
Przechwytuje obraz z kamery, wykrywa twarz i generuje animowaną postać patyczaka.
Dodano rozszerzone debugowanie i wizualizację wykrywania twarzy.
"""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp

# Importy z uproszczonych modułów w katalogu drawing
from drawing.stick_figure_renderer import StickFigureRenderer


class StickFigureWebcam:
    """
    Główna klasa aplikacji, która łączy wszystkie komponenty i zarządza przepływem danych.
    Uproszczona wersja koncentrująca się na wykrywaniu twarzy i rysowaniu patyczaka na środku ekranu.
    Dodano rozszerzone debugowanie i wizualizację wykrywania twarzy.
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
        show_face_mesh: bool = True  # Nowa opcja - pokazywanie siatki twarzy
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
        """
        # Konfiguracja
        self.width = width
        self.height = height
        self.fps = fps
        self.debug = debug
        self.show_preview = show_preview
        self.flip_camera = flip_camera
        self.use_virtual_camera = use_virtual_camera
        self.show_face_mesh = show_face_mesh

        # Flagi stanu
        self.running = False
        self.paused = False
        self.show_chair = False

        print(f"Inicjalizacja Stick Figure Webcam ({width}x{height} @ {fps}FPS)")

        # Inicjalizacja komponentów
        try:
            # Inicjalizacja kamery
            print("Inicjalizacja kamery...")
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)

            if not self.camera.isOpened():
                raise Exception("Nie można otworzyć kamery")

            # Inicjalizacja wirtualnej kamery (jeśli wymagana)
            self.virtual_camera_ready = False
            if use_virtual_camera:
                try:
                    import pyvirtualcam
                    print("Inicjalizacja wirtualnej kamery...")
                    # Poprawione argumenty dla pyvirtualcam (usuń device_type)
                    self.virtual_camera = pyvirtualcam.Camera(
                        width=width,
                        height=height,
                        fps=fps
                    )
                    self.virtual_camera_ready = True
                    print(f"Wirtualna kamera gotowa: {self.virtual_camera.device}")
                except Exception as e:
                    print(f"Nie można zainicjalizować wirtualnej kamery: {e}")
                    print("Aplikacja będzie działać bez wirtualnej kamery.")

            # Inicjalizacja detektora twarzy MediaPipe
            print("Inicjalizacja detektora twarzy...")
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

            # Inicjalizacja renderera stick figure
            print("Inicjalizacja renderera stick figure...")
            self.renderer = StickFigureRenderer(
                canvas_width=width,
                canvas_height=height,
                line_thickness=3,
                head_radius_factor=0.075,
                bg_color=(255, 255, 255),  # Białe tło
                figure_color=(0, 0, 0),  # Czarny patyczak
                smooth_factor=0.3
                # Usunięto parametr logger
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

            print("Stick Figure Webcam zainicjalizowany pomyślnie")

        except Exception as e:
            print(f"Błąd podczas inicjalizacji: {str(e)}")
            raise

    def run(self):
        """
        Uruchamia główną pętlę aplikacji.
        """
        self.running = True
        print("Uruchamianie głównej pętli aplikacji")
        print(
            "Naciśnij 'q' aby zakończyć, 'p' aby wstrzymać/wznowić, 's' aby zmienić nastrój, 'c' aby włączyć/wyłączyć krzesło")
        print("Naciśnij 'm' aby włączyć/wyłączyć wizualizację siatki twarzy, 'd' aby włączyć/wyłączyć tryb debugowania")

        try:
            # Główna pętla aplikacji
            while self.running:
                # Obliczanie FPS
                current_time = time.time()
                if current_time - self.fps_timer >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_timer)
                    self.fps_timer = current_time
                    self.fps_counter = 0

                # W trybie pauzy tylko sprawdzamy klawisze
                if self.paused:
                    self._handle_keys()
                    time.sleep(0.05)  # Zmniejszamy obciążenie CPU
                    continue

                # 1. Pobieranie klatki z kamery
                ret, frame = self.camera.read()

                if not ret:
                    print("Nie udało się odczytać klatki z kamery")
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

                # Kopiowanie obrazu do wyświetlenia z zaznaczeniami
                debug_frame = frame.copy()

                # Przetworzenie wyników detekcji twarzy
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
                    if face_data and "expressions" in face_data:
                        self.last_expression_values = face_data["expressions"]

                # 4. Renderowanie stick figure (zawsze na środku ekranu)
                output_image = self.renderer.render(face_data, self.show_chair)

                # 5. Wysyłanie obrazu do wirtualnej kamery
                if self.virtual_camera_ready:
                    # Konwersja BGR -> RGB dla pyvirtualcam
                    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    self.virtual_camera.send(output_rgb)

                # 6. Wyświetlanie podglądu
                if self.show_preview:
                    self._show_preview(debug_frame, output_image)

                # 7. Obsługa klawiszy
                self._handle_keys()

                # Aktualizacja liczników
                self.frame_count += 1
                self.fps_counter += 1

                # Limit FPS
                elapsed_time = time.time() - current_time
                target_time = 1.0 / self.fps

                if elapsed_time < target_time:
                    time.sleep(target_time - elapsed_time)

        except KeyboardInterrupt:
            print("Przerwanie przez użytkownika (Ctrl+C)")
        except Exception as e:
            print(f"Błąd w głównej pętli: {str(e)}")
            raise
        finally:
            self._cleanup()

    def _process_face_landmarks(self, face_landmarks):
        """
        Przetwarza punkty charakterystyczne twarzy z MediaPipe na format używany przez nasz renderer.

        Args:
            face_landmarks: Punkty charakterystyczne twarzy z MediaPipe

        Returns:
            dict: Słownik z danymi twarzy
        """
        # Analizujemy kluczowe punkty twarzy, aby określić wyrazy mimiczne

        # Domyślne wartości
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

            # Zwiększamy czułość na zmiany (mnożymy przez 10 zamiast 5)
            height_diff = center_mouth.y - corner_height_avg

            # Debug - wypisujemy wartości różnicy wysokości
            if self.debug and self.frame_count % 30 == 0:
                print(f"Różnica wysokości kącików ust: {height_diff:.6f}")

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
                print(f"Wartość smile: {smile:.2f}, mouth_open: {mouth_open:.2f}")

        except Exception as e:
            if self.debug:
                print(f"Błąd podczas analizy mimiki twarzy: {str(e)}")

        # Tworzymy słownik z danymi twarzy
        return {
            "has_face": True,
            "expressions": {
                "mouth_open": mouth_open,
                "smile": smile,
                "left_eye_open": 1.0,
                "right_eye_open": 1.0,
            }
        }

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

            # Dodajemy informację o stanie krzesła
            chair_text = "Krzeslo: ON" if self.show_chair else "Krzeslo: OFF (klawisz 'c')"
            cv2.putText(
                original_frame,
                chair_text,
                (10, self.height - 50),
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
            print(f"Błąd podczas wyświetlania podglądu: {str(e)}")

    def _handle_keys(self):
        """
        Obsługuje wciśnięcia klawiszy.
        """
        wait_time = 10 if self.paused else 1
        key = cv2.waitKey(wait_time) & 0xFF

        if key == 27 or key == ord('q'):  # ESC lub q - wyjście
            self.running = False
            print("Wyjście z aplikacji")

        elif key == ord('p'):  # p - pauza/wznowienie
            self.paused = not self.paused
            status = "Wstrzymano" if self.paused else "Wznowiono"
            print(f"{status} przetwarzanie")

        elif key == ord('f'):  # f - przełączenie odbicia poziomego
            self.flip_camera = not self.flip_camera
            print(f"Odbicie poziome: {'włączone' if self.flip_camera else 'wyłączone'}")

        elif key == ord('c'):  # c - włączenie/wyłączenie krzesła
            self.show_chair = not self.show_chair
            print(f"Krzesło: {'włączone' if self.show_chair else 'wyłączone'}")

        elif key == ord('m'):  # m - włączenie/wyłączenie wizualizacji siatki twarzy
            self.show_face_mesh = not self.show_face_mesh
            print(f"Wizualizacja siatki twarzy: {'włączona' if self.show_face_mesh else 'wyłączona'}")

        elif key == ord('d'):  # d - włączenie/wyłączenie trybu debugowania
            self.debug = not self.debug
            print(f"Tryb debugowania: {'włączony' if self.debug else 'wyłączony'}")

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
            print(f"Zmieniono nastrój na: {new_mood}")

    def _cleanup(self):
        """
        Zwalnia zasoby przed zakończeniem.
        """
        print("Zamykanie zasobów...")

        try:
            # Zamykanie kamery
            if hasattr(self, 'camera'):
                self.camera.release()

            # Zamykanie wirtualnej kamery
            if hasattr(self, 'virtual_camera_ready') and self.virtual_camera_ready:
                self.virtual_camera.close()

            # Zamykanie MediaPipe Face Mesh
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()

            # Zamykanie okien OpenCV
            cv2.destroyAllWindows()

            print("Wszystkie zasoby zamknięte pomyślnie")

        except Exception as e:
            print(f"Błąd podczas zwalniania zasobów: {str(e)}")


def parse_arguments():
    """
    Parsuje argumenty linii poleceń.
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
    parser.add_argument("--chair", action="store_true",
                        help="Włącza pokazywanie krzesła")
    parser.add_argument("--no-face-mesh", action="store_true",
                        help="Wyłącza wizualizację siatki twarzy")

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
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)

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
            show_face_mesh=not args.no_face_mesh
        )

        # Ustawienie opcji krzesła
        app.show_chair = args.chair

        app.run()
    except Exception as e:
        print(f"Krytyczny błąd aplikacji: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
