#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stick Figure Webcam - Main entry point.

A webcam application that converts your video feed into an animated
stick figure character, perfect for video calls and streaming.
"""

import argparse
import sys

from src.app.config_manager import ConfigurationManager
from src.app.controller import ApplicationController
from src.utils.custom_logger import CustomLogger
from src.utils.logging_config import setup_logger


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Stick Figure Webcam - Transform yourself into an animated stick figure"
    )

    parser.add_argument("-c", "--camera", type=int, default=None, help="Camera ID (default: 0)")

    parser.add_argument("-w", "--width", type=int, default=None, help="Video width (default: 640)")

    parser.add_argument(
        "-H", "--height", type=int, default=None, help="Video height (default: 480)"
    )

    parser.add_argument("-f", "--fps", type=int, default=None, help="Target FPS (default: 30)")

    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("--no-preview", action="store_true", help="Disable preview windows")

    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")

    return parser.parse_args()


def apply_cli_overrides(config: ConfigurationManager, args):
    """Apply command line argument overrides to configuration.

    Args:
        config: Configuration manager
        args: Parsed arguments
    """
    if args.camera is not None:
        config.set("camera.id", args.camera)

    if args.width is not None:
        config.set("camera.width", args.width)

    if args.height is not None:
        config.set("camera.height", args.height)

    if args.fps is not None:
        config.set("camera.fps", args.fps)

    if args.debug:
        config.set("app.debug", True)

    if args.no_preview:
        config.set("app.show_preview", False)


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logger(CustomLogger, debug=args.debug)

    if logger is None:
        print("ERROR: Failed to initialize logger")
        sys.exit(1)

    logger.info("Main", "=== Stick Figure Webcam Starting ===")

    try:
        # Load configuration
        config = ConfigurationManager(config_path=args.config, logger=logger)

        # Apply CLI overrides
        apply_cli_overrides(config, args)

        # Validate configuration
        if not config.validate():
            logger.warning("Main", "Configuration validation failed, using defaults")

        # Create application controller
        app = ApplicationController(config, logger)

        # Run application
        app.run()

        logger.info("Main", "=== Stick Figure Webcam Shutdown Complete ===")

    except KeyboardInterrupt:
        logger.info("Main", "Application interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.critical("Main", f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
