# Test Suite Documentation

This directory contains the test suite for Stick Figure Webcam.

## Test Coverage

Current test coverage is aimed at **70%+** of the codebase, focusing on critical functionality.

## Test Structure

```
tests/
├── drawing/           # Tests for rendering components
│   ├── test_face_renderer.py
│   ├── test_hand_tracking.py
│   ├── test_stick_figure.py
│   └── test_drawing_utils.py
├── pose/              # Tests for pose detection
│   ├── test_pose_detector.py
│   ├── test_posture_analyzer.py
│   └── test_partial_visibility.py
├── utils/             # Tests for utility modules
│   ├── test_custom_logger.py
│   ├── test_performance.py
│   ├── test_system_check.py
│   └── test_theme_utils.py
├── camera/            # Tests for camera modules
│   ├── test_camera_capture.py
│   └── test_virtual_camera.py
├── app/               # Tests for app modules
│   ├── test_config_manager.py
│   ├── test_video_pipeline.py
│   └── test_integration_pipeline.py
└── lighting/          # Tests for adaptive lighting
    └── test_adaptive_colors.py
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run with coverage report

```bash
pytest --cov=src --cov-report=html
```

### Run specific test file

```bash
pytest tests/pose/test_posture_analyzer.py
```

### Run specific test

```bash
pytest tests/pose/test_posture_analyzer.py::TestPostureAnalyzer::test_analyze_posture_sitting
```

### Run with verbose output

```bash
pytest -v
```

## Test Categories

### Unit Tests

Test individual components in isolation:

- `test_face_renderer.py` - Face rendering logic
- `test_pose_detector.py` - MediaPipe pose detection
- `test_camera_capture.py` - Camera operations
- `test_adaptive_colors.py` - Lighting adaptation

### Integration Tests

Test component interactions:

- `test_integration_pipeline.py` - Complete video pipeline
- `test_hand_tracking.py` - Hand tracking with stick figure

### Edge Case Tests

Test boundary conditions:

- `test_partial_visibility.py` - Limited body visibility scenarios

## Writing New Tests

Follow these guidelines (from CONTRIBUTING.md):

1. **Test important functionality** - Focus on business logic, not getters/setters
2. **Use descriptive names** - `test_analyze_posture_sitting` not `test_1`
3. **Keep tests focused** - One test per behavior
4. **Mock external dependencies** - Use `unittest.mock` for I/O, network, etc.
5. **Use fixtures from conftest.py** - Reuse common test data

### Example Test Structure

```python
import unittest
from unittest.mock import MagicMock

from src.module.component import Component


class TestComponent(unittest.TestCase):
    """Tests for Component class."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()
        self.component = Component(logger=self.mock_logger)

    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = self.component.process(input_data)

        # Assert
        self.assertEqual(result.value, expected_value)
        self.mock_logger.info.assert_called_once()
```

## Test Coverage Goals

Target coverage by module:

- **pose/** - 80%+ (critical for core functionality)
- **drawing/** - 75%+ (visual output)
- **camera/** - 70%+ (I/O operations)
- **app/** - 70%+ (orchestration)
- **utils/** - 65%+ (helpers)
- **lighting/** - 60%+ (nice-to-have feature)

## Common Mocking Patterns

### Mocking OpenCV VideoCapture

```python
mock_cap = MagicMock()
mock_cap.isOpened.return_value = True
mock_cap.read.return_value = (True, test_frame)
mock_cap.get.side_effect = lambda x: {3: 640, 4: 480}.get(int(x), 0)

with patch('cv2.VideoCapture', return_value=mock_cap):
# Your test code
```

### Mocking MediaPipe

```python
mock_results = MagicMock()
mock_results.pose_landmarks = create_mock_landmarks()

mock_pose = MagicMock()
mock_pose.process.return_value = mock_results

with patch('mediapipe.solutions.pose.Pose', return_value=mock_pose):
# Your test code
```

### Mocking Logger

```python
mock_logger = MagicMock()
component = Component(logger=mock_logger)

# Verify logging happened
mock_logger.info.assert_called_once()
mock_logger.error.assert_called_with("Expected message")
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_landmarks` - Standard 33-point pose data
- `sample_image` - 640x480 test image
- `mock_logger` - Pre-configured mock logger

## Continuous Integration

Tests run automatically on:

- Every push to main branch
- Every pull request
- Nightly builds

CI fails if:

- Any test fails
- Coverage drops below 70%
- New code isn't tested

## Troubleshooting

### Tests fail with ImportError

Make sure you're in the project root and have installed dev dependencies:

```bash
pip install -e ".[dev]"
```

### Mock not working

Ensure you're patching at the right location:

```python
# Patch where it's used, not where it's defined
@patch('src.module.using_it.ExternalClass')
```

### Flaky tests

Add delays or use deterministic mocking:

```python
with patch('time.time', return_value=1234567890):
# Deterministic time
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)
- [Project CONTRIBUTING.md](../CONTRIBUTING.md)
