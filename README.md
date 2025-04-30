# Stick Figure Webcam

## Opis projektu

Aplikacja w Pythonie, która za pomocą kamery internetowej rejestruje ruchy użytkownika i jego pozę, a następnie zamienia je na animowaną postać stick figure (ludzika z kresek) na białym tle. Program wykrywa, czy użytkownik siedzi na krześle czy stoi, i odpowiednio dostosowuje animację. Wygenerowany obraz jest dostępny jako wirtualna kamera, którą można wykorzystać w aplikacjach takich jak Discord, Zoom, Teams czy inne komunikatory wideo.

## Cele projektu

- Stworzenie alternatywnego, anonimowego sposobu uczestniczenia w wideokonferencjach
- Zredukowanie wymagań dotyczących przepustowości łącza internetowego
- Dodanie elementu rozrywki do wideokonferencji
- Praktyczne wykorzystanie detekcji pozy człowieka i przetwarzania obrazu w czasie rzeczywistym

## Wymagania systemowe

### Wymagania programowe
- Python 3.8 lub nowszy (najlepiej 3.10)
- Biblioteki: OpenCV, MediaPipe, NumPy, PyVirtualCam
- Dostęp do kamery internetowej

### Obsługiwane systemy operacyjne
- Windows: wymaga OBS Virtual Camera lub podobnej aplikacji
- macOS: wymaga OBS Virtual Camera + odpowiedniego pluginu
- Linux: wymaga modułu v4l2loopback

## Instalacja

1. Sklonuj repozytorium:
    ```bash
    git clone https://github.com/philornot/StickfigureWebcam.git
    cd StickfigureWebcam
    ```

2. Utwórz i aktywuj wirtualne środowisko (opcjonalnie, ale zalecane):
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate
    ```

3. Zainstaluj wymagane biblioteki:
    ```bash
    pip install -r requirements.txt
    ```

4. Zainstaluj dodatkowe oprogramowanie:
   - Windows: Zainstaluj [OBS Studio](https://obsproject.com/) i uruchom Virtual Camera
   - macOS: Zainstaluj [OBS Studio](https://obsproject.com/) i [OBS Virtual Camera Plugin](https://github.com/johnboiles/obs-mac-virtualcam)
   - Linux: Zainstaluj v4l2loopback:
     ```bash
     sudo apt-get install v4l2loopback-dkms
     sudo modprobe v4l2loopback
     ```

## Użycie

1. Uruchom aplikację:
    ```bash
    python src/main.py
    ```

2. Aplikacja uruchomi się, aktywując kamerę internetową i rozpoczynając detekcję pozy.

3. W aplikacjach do wideokonferencji (np. Discord, Zoom, Teams) wybierz "Stick Figure Webcam" jako źródło wideo.

4. Dostosuj ustawienia w pliku `src/config/settings.py` według potrzeb.

### Testy
Aby uruchomić testy jednostkowe, uruchom
```commandline
python -m pytest
```

## Funkcje

- Detekcja pozy człowieka w czasie rzeczywistym
- Automatyczne wykrywanie pozycji siedzącej/stojącej
- Rysowanie stick figure odwzorowującego ruch użytkownika
- Przekazywanie generowanego obrazu jako wirtualnej kamery
- Niskie obciążenie procesora i sieci

## Rozwiązywanie problemów

- **Problem:** Nie wykrywa poprawnie pozy
  - **Rozwiązanie:** Upewnij się, że jesteś dobrze oświetlony i widoczny w kadrze kamery

- **Problem:** Wirtualna kamera nie jest widoczna w aplikacjach
  - **Rozwiązanie:** Upewnij się, że OBS Virtual Camera jest uruchomiona, lub że moduł v4l2loopback jest poprawnie załadowany

- **Problem:** Aplikacja działa wolno
  - **Rozwiązanie:** Zmniejsz rozdzielczość kamery w ustawieniach lub obniż docelowe FPS

## Rozwijanie projektu

Możliwe rozszerzenia:

1. **Ekspresje twarzy** - dodanie detekcji emocji
2. **Customizacja wyglądu** - możliwość zmiany stylu stick figure
3. **Detekcja gestów** - rozpoznawanie specjalnych gestów (np. machanie, kciuk w górę)
4. **Filtrowanie drgań** - wygładzanie ruchów stick figure
5. **Tła i rekwizyty** - dodanie interaktywnych elementów otoczenia

## Licencja

Ten projekt jest objęty licencją [MIT](LICENSE).

## Autor

philornot