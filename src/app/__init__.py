#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application core modules.

This package contains the main application components:
- ApplicationController: Main application lifecycle manager
- VideoPipeline: Video processing pipeline
- UIManager: User interface management
- ConfigurationManager: Configuration handling
- DependencyContainer: Dependency injection
"""

from src.app.config_manager import ConfigurationManager
from src.app.controller import ApplicationController
from src.app.dependencies import DependencyContainer, create_container
from src.app.ui_manager import UIManager
from src.app.video_pipeline import VideoPipeline

__all__ = [
    "ApplicationController",
    "VideoPipeline",
    "UIManager",
    "ConfigurationManager",
    "DependencyContainer",
    "create_container",
]
