"""
Stickfigure Webcam - Main Entry Point

This is the main entry point for the Stickfigure Webcam application.
It automatically detects whether GUI mode is available and falls back
to CLI mode if necessary.

Usage:
    python gui_main.py              # Auto-detect (prefer GUI)
    python gui_main.py --no-gui     # Force CLI mode
    python gui_main.py --cli        # Force CLI mode (alias)
"""

import sys


def main():
    """
    Main entry point for Stickfigure Webcam.

    Checks for GUI availability and command-line arguments,
    then launches the appropriate mode (GUI or CLI).
    """
    # Check for CLI mode flags
    force_cli = '--no-gui' in sys.argv or '--cli' in sys.argv

    if force_cli:
        # User explicitly requested CLI mode
        print("[Launcher] CLI mode requested")
        run_cli_mode()
    else:
        # Try GUI mode, fall back to CLI if unavailable
        try:
            import PyQt6
            print("[Launcher] GUI mode available, starting...")
            run_gui_mode()
        except ImportError:
            print("[Launcher] GUI not available (PyQt6 not installed)")
            print("[Launcher] Install with: pip install PyQt6")
            print("[Launcher] Falling back to CLI mode...")
            print()
            run_cli_mode()


def run_gui_mode():
    """Launch the application in GUI mode."""
    try:
        from gui.app import run_gui
        sys.exit(run_gui())
    except Exception as e:
        print(f"[Launcher] GUI mode failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_cli_mode():
    """Launch the application in CLI mode."""
    try:
        from main import main as cli_main
        cli_main()
    except Exception as e:
        print(f"[Launcher] CLI mode failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
