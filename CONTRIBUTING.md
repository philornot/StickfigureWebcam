# Contributing to Stick Figure Webcam

Hii... are you a human? Anyways:

## Quick Start

### Setup

```bash
# Clone it
git clone https://github.com/philornot/StickfigureWebcam.git
cd StickfigureWebcam

# Virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install everything
pip install -e ".[dev]"

# Setup pre-commit (auto-formats on commit)
pre-commit install
```

## Development Workflow

```bash
# Run the thing
python src/main.py

# Format code (or let pre-commit do it automatically)
black src tests

# Run tests
pytest

# Run specific test
pytest tests/pose/test_posture_analyzer.py
```

## Code Style

I'm using **Black** for formatting (line length: 100), so just let it do its thing.

For docstrings, I try to follow Google style, but honestly as long as it's clear what the function does, I'm happy:

```python
def do_something(x: int, y: str) -> bool:
    """Does something with x and y.

    Args:
        x: Some number
        y: Some string

    Returns:
        True if it worked, False otherwise
    """
    return True
```

Type hints are nice to have but not required everywhere. Use them where it makes sense.

## Project Structure

```
src/
├── camera/        # Camera capture and virtual camera stuff
├── drawing/       # Stick figure rendering (main logic here)
├── pose/          # MediaPipe pose detection
├── utils/         # Helper stuff (logging, performance monitoring)
└── main.py        # Entry point

tests/             # Mirror src/ structure
```

## Testing

Tests are in the `tests/` folder. I try to test the important stuff but let's be real, 100% coverage is overkill for a
fun project.

Run tests:

```bash
pytest
```

## Pre-commit Hooks

They auto-run when you commit. They just:

- Remove trailing whitespace
- Fix line endings
- Format code with Black
- Check for big files

If you need to skip them (not recommended but sometimes necessary):

```bash
SKIP=black git commit -m "Quick fix"
```

## Making Changes

1. Create a branch: `git checkout -b feature/cool-thing`
2. Make your changes
3. Test it works: `pytest`
4. Commit (pre-commit will auto-format)
5. Push and make a PR

No strict commit message format required. Just make it clear what you did.

## File Organization

Try to keep files under 500 lines. If something gets too big, split it up.

Comments should explain *why*, not *what*. The code already shows what it does.

## Documentation

If you add a new feature:

- Update the README
- Add some docstrings
- Maybe write a test

## Language

- Code, comments, and docstrings: **English**
- UI messages for users: **Polish is fine**, but I will make everything translatable so in the end it doesn't matter.
