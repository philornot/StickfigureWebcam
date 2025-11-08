# Contributing to Stick Figure Webcam

Thank you for your interest in contributing to Stick Figure Webcam! This document provides guidelines and instructions for developers.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Development Setup

### Prerequisites

- Python 3.10 or 3.11
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/philornot/StickfigureWebcam.git
   cd StickfigureWebcam
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Quality Standards

### Style Guidelines

We follow these style guides:
- **PEP 8** for Python code style
- **Google Style** for docstrings
- **Black** for code formatting (line length: 100)
- **isort** for import sorting

### Code Formatting

Formatting code isn't necessary, but it doesn't hurt to do:

```bash
# Format code
make format

# Or manually
black src tests
isort src tests
```

### Type Hints

All public functions and methods should include type hints:

```python
def process_frame(
    frame: np.ndarray,
    landmarks: List[Tuple[float, float, float, float]]
) -> Dict[str, Any]:
    """Process a video frame with detected landmarks.

    Args:
        frame: Input frame as NumPy array
        landmarks: List of detected pose landmarks

    Returns:
        Dictionary containing processing results
    """
    ...
```

### Docstring Format

We use Google-style docstrings for all public APIs:

```python
def analyze_posture(
    landmarks: List[Tuple[float, float, float, float]],
    frame_height: int,
    frame_width: int
) -> Dict[str, Any]:
    """Analyze user posture based on detected landmarks.

    This function analyzes the position of key body points to determine
    whether the user is sitting or standing.

    Args:
        landmarks: List of 33 pose landmarks in format (x, y, z, visibility)
        frame_height: Height of the video frame in pixels
        frame_width: Width of the video frame in pixels

    Returns:
        Dictionary containing:
            - is_sitting: Boolean indicating sitting posture
            - confidence: Float confidence score (0.0-1.0)
            - posture: String describing the posture

    Raises:
        ValueError: If landmarks list has incorrect length

    Example:
        >>> landmarks = [(0.5, 0.5, 0.0, 0.9)] * 33
        >>> result = analyzer.analyze_posture(landmarks, 480, 640)
        >>> print(result['is_sitting'])
        True
    """
    ...
```

### File Size Limit

Keep Python files under 500 lines. If a file grows too large:
1. Extract related functionality into separate modules
2. Create helper functions
3. Consider splitting classes with multiple responsibilities

## Pre-commit Hooks

Pre-commit hooks automatically check code quality before each commit. They will:

- Remove trailing whitespace
- Fix end-of-file issues
- Check YAML syntax
- Prevent large files (>500KB)
- Sort imports with isort
- Format code with black
- Check style with flake8
- Type check with mypy
- Validate docstrings with pydocstyle
- Ensure files don't exceed 500 lines

### Running Hooks Manually

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Skip hooks temporarily (not recommended)
SKIP=flake8,mypy git commit -m "Temporary commit"
```

### Updating Hook Versions

```bash
pre-commit autoupdate
```

## Testing

### Running Tests

```bash
# Run all tests
make test
# or
pytest

# Run specific test file
pytest tests/pose/test_posture_analyzer.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Coverage Requirements

- Minimum 70% overall coverage
- All public methods must have tests
- Critical paths must have integration tests

### Writing Tests

Place tests in the `tests/` directory mirroring the `src/` structure:

```
tests/
├── pose/
│   ├── test_pose_detector.py
│   └── test_posture_analyzer.py
└── drawing/
    └── test_stick_figure.py
```

Test file template:

```python
import unittest
from src.module import MyClass


class TestMyClass(unittest.TestCase):
    """Tests for MyClass functionality."""

    def setUp(self):
        """Initialize test fixtures."""
        self.instance = MyClass()

    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        result = self.instance.method()
        self.assertEqual(result, expected_value)
```

## Documentation

### Inline Comments

- Use English for all comments and docstrings
- Polish is acceptable only for user-facing GUI messages
- Comment the "why", not the "what"
- Keep comments concise and up-to-date

### API Documentation

Public APIs should be fully documented with:
- Clear description
- All parameters with types
- Return value description
- Possible exceptions
- Usage examples

### README Updates

Update README.md when adding:
- New features
- New dependencies
- Changed system requirements
- New configuration options

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Ensure quality checks pass:**
   ```bash
   make lint
   make type-check
   make test
   ```

4. **Push and create pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

If you want to, you can follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks
