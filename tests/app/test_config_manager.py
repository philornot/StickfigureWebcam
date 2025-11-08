#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for configuration manager."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from src.app.config_manager import ConfigurationManager


class TestConfigurationManager(unittest.TestCase):
    """Tests for ConfigurationManager class."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_initialization_default(self):
        """Test initialization with default config."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        # Should have default values
        self.assertEqual(config.get("camera.width"), 640)
        self.assertEqual(config.get("camera.height"), 480)

    def test_get_nested_value(self):
        """Test getting nested configuration value."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        value = config.get("camera.width")
        self.assertEqual(value, 640)

        # Non-existent key with default
        value = config.get("non.existent.key", "default")
        self.assertEqual(value, "default")

    def test_set_nested_value(self):
        """Test setting nested configuration value."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        config.set("camera.width", 1280)

        self.assertEqual(config.get("camera.width"), 1280)

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config1 = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        config1.set("camera.width", 1920)
        config1.set("camera.height", 1080)
        config1.save()

        # Load in new instance
        config2 = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        self.assertEqual(config2.get("camera.width"), 1920)
        self.assertEqual(config2.get("camera.height"), 1080)

    def test_validate_valid_config(self):
        """Test validating valid configuration."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        result = config.validate()

        self.assertTrue(result)

    def test_validate_invalid_width(self):
        """Test validating invalid width."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        config.set("camera.width", 100)  # Too small

        result = config.validate()

        self.assertFalse(result)

    def test_validate_invalid_fps(self):
        """Test validating invalid FPS."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        config.set("camera.fps", 200)  # Too high

        result = config.validate()

        self.assertFalse(result)

    def test_get_camera_config(self):
        """Test getting camera configuration."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        cam_config = config.get_camera_config()

        self.assertIn("width", cam_config)
        self.assertIn("height", cam_config)
        self.assertIn("fps", cam_config)

    def test_get_processing_config(self):
        """Test getting processing configuration."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        proc_config = config.get_processing_config()

        self.assertIn("line_thickness", proc_config)
        self.assertIn("bg_color", proc_config)

    def test_deep_merge(self):
        """Test deep merging of configurations."""
        config = ConfigurationManager(config_path=self.config_file, logger=self.mock_logger)

        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 4}

        result = config._deep_merge(base, override)

        self.assertEqual(result["a"]["b"], 10)  # Overridden
        self.assertEqual(result["a"]["c"], 2)  # Preserved
        self.assertEqual(result["d"], 3)  # Preserved
        self.assertEqual(result["e"], 4)  # Added


if __name__ == "__main__":
    unittest.main()
