# Stick Figure Webcam 🎭

<div align="center">

*[English version (README-en.md)](README-en.md)*

![Wersja Pythona](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue?logo=python&logoColor=white)
![Licencja](https://img.shields.io/badge/licencja-MIT-green)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![Styl kodu: black](https://img.shields.io/badge/styl%20kodu-black-000000.svg)](https://github.com/psf/black)
[![Importy: isort](https://img.shields.io/badge/importy-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)
[![Typowanie statyczne: mypy](https://img.shields.io/badge/typowanie%20statyczne-mypy-blue)](http://mypy-lang.org/)

<img src="stick-figure-animated.svg" width="250" height="250" alt="Animacja Stick Figure">

_**Bądź ludzikiem z kresek na swojej następnej wideokonferencji!**_

</div>

## 📋 Opis projektu

Aplikacja w Pythonie, która za pomocą kamery internetowej rejestruje ruchy użytkownika i jego pozę, a następnie zamienia
je na animowaną postać stick figure (ludzika z kresek) na białym tle. Program wykrywa, czy użytkownik siedzi na krześle
czy stoi, i odpowiednio dostosowuje animację. Wygenerowany obraz jest dostępny jako wirtualna kamera, którą można
wykorzystać w aplikacjach takich jak Discord, Zoom, Teams czy inne komunikatory wideo.

### ✨ Główne funkcje

- 🕺 Detekcja pozy człowieka w czasie rzeczywistym
- 🪑 Automatyczne wykrywanie pozycji siedzącej/stojącej
- 🖌️ Renderowanie stick figure odzwierciedlającego twoje ruchy
- 🎥 Wyjście obrazu przez wirtualną kamerę do aplikacji wideo
- 🚀 Niskie obciążenie procesora i przepustowości łącza

## 🎯 Cele projektu

- Stworzenie alternatywnego, anonimowego sposobu uczestniczenia w wideokonferencjach
- Zredukowanie wymagań dotyczących przepustowości łącza internetowego
- Dodanie elementu rozrywki do wideokonferencji
- Nauczenie się podstaw detekcji pozy człowieka i przetwarzania obrazu w czasie rzeczywistym

## 🖥️ Wymagania systemowe

### Wymagania programowe

- Python 3.8 lub nowszy (zalecany 3.10)
- Biblioteki: OpenCV, MediaPipe, NumPy, PyVirtualCam
- Dostęp do kamery internetowej

### Obsługiwane systemy operacyjne

- **Windows**: wymaga OBS Virtual Camera lub podobnej aplikacji
- **macOS**: wymaga OBS Virtual Camera + odpowiedniego pluginu
- **Linux**: wymaga modułu v4l2loopback

## 🔧 Instalacja

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
    # Dla użytkowników: zainstaluj tylko aplikację
    pip install -e .

    # Dla deweloperów: zainstaluj z narzędziami programistycznymi
    pip install -e ".[dev]"
    ```

4. Zainstaluj dodatkowe oprogramowanie:
    - **Windows**: Zainstaluj [OBS Studio](https://obsproject.com/) i uruchom Virtual Camera
    - **macOS**: Zainstaluj [OBS Studio](https://obsproject.com/)
      i [OBS Virtual Camera Plugin](https://github.com/johnboiles/obs-mac-virtualcam)
    - **Linux**: Zainstaluj v4l2loopback:
      ```bash
      sudo apt-get install v4l2loopback-dkms
      sudo modprobe v4l2loopback
      ```

## 🚀 Użycie

### Uruchamianie aplikacji

```bash
# Za pomocą modułu Python
python -m src.main

# Używając make (jeśli masz zainstalowane Make)
make run

# Z opcjami linii poleceń
python src/main.py --width 640 --height 480 --fps 30 --debug
```

Aplikacja uruchomi się, aktywuje kamerę internetową i rozpocznie detekcję pozy. W aplikacjach do wideokonferencji (
Discord, Zoom, Teams) wybierz "Stick Figure Webcam" jako źródło wideo.

### Sterowanie z klawiatury

| Klawisz       | Funkcja                                |
|---------------|----------------------------------------|
| `q` lub `ESC` | Wyjście z aplikacji                    |
| `p`           | Wstrzymanie/wznowienie przetwarzania   |
| `d`           | Przełączanie trybu debugowania         |
| `f`           | Przełączanie odbicia poziomego         |
| `t`           | Wysłanie wzoru testowego               |
| `r`           | Reset wirtualnej kamery                |
| `l`           | Przełączanie adaptacyjnego oświetlenia |

### Komendy dla deweloperów

```bash
# Uruchomienie testów
make test
python -m pytest

# Sprawdzenie stylu kodu
make lint

# Sprawdzenie typów
make type-check

# Formatowanie kodu
make format

# Konfiguracja środowiska deweloperskiego
make dev-setup
```

## 📁 Struktura projektu

```
StickfigureWebcam/
├── src/                  # Kod źródłowy
│   ├── camera/           # Przechwytywanie obrazu i wirtualna kamera
│   ├── config/           # Ustawienia aplikacji
│   ├── drawing/          # Renderowanie stick figure
│   ├── lighting/         # Adaptacyjne oświetlenie
│   ├── pose/             # Detekcja i analiza pozy
│   └── utils/            # Funkcje pomocnicze
├── tests/                # Testy jednostkowe
├── pyproject.toml        # Konfiguracja projektu
├── requirements.txt      # Zależności runtime
├── requirements-dev.txt  # Zależności deweloperskie
└── Makefile              # Zadania deweloperskie
```

## 🛠️ Pliki konfiguracyjne

Projekt zawiera kilka plików konfiguracyjnych dla nowoczesnego developmentu w Pythonie:

- **pyproject.toml**: Główna konfiguracja projektu (metadane, zależności, ustawienia narzędzi)
- **tox.ini**: Konfiguracja testów w wielu środowiskach
- **.pre-commit-config.yaml**: Hooki Git dla jakości kodu
- **mypy.ini**: Konfiguracja sprawdzania typów
- **.flake8**: Reguły sprawdzania stylu kodu
- **.editorconfig**: Ustawienia stylu kodu niezależne od edytora
- **Makefile**: Wygodne komendy deweloperskie

## 🔍 Rozwiązywanie problemów

- **Problem:** Nie wykrywa poprawnie pozy
    - **Rozwiązanie:** Upewnij się, że jesteś dobrze oświetlony i widoczny w kadrze kamery

- **Problem:** Wirtualna kamera nie jest widoczna w aplikacjach
    - **Rozwiązanie:** Upewnij się, że OBS Virtual Camera jest uruchomiona, lub że moduł v4l2loopback jest poprawnie
      załadowany

- **Problem:** Aplikacja działa wolno
    - **Rozwiązanie:** Zmniejsz rozdzielczość kamery w ustawieniach lub obniż docelowe FPS

## 🧩 Możliwe rozszerzenia

Możliwe rozszerzenia dla projektu:

1. **Ekspresje twarzy** - dodanie detekcji emocji i animacja
2. **Customizacja wyglądu** - możliwość zmiany stylu stick figure
3. **Detekcja gestów** - rozpoznawanie specjalnych gestów (np. machanie, kciuk w górę)
4. **Filtrowanie drgań** - wygładzanie ruchów stick figure
5. **Tła i rekwizyty** - dodanie interaktywnych elementów otoczenia

## 📄 Licencja

Ten projekt jest objęty licencją [MIT](LICENSE).

## 👤 Autor

philornot
