"""
Persistent configuration storage module.

This module handles saving and loading user preferences to/from a JSON file
in the user's configuration directory with automatic debouncing and
fallback to default values if configuration is corrupted.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class PersistentConfig:
    """
    Thread-safe persistent configuration manager.

    Automatically saves configuration changes to disk with debouncing
    to prevent excessive writes. Provides fallback to defaults if
    configuration file is corrupted or missing.
    """

    def __init__(self, app_name: str = "stickfigure_webcam",
                 config_filename: str = "config.json",
                 debounce_seconds: float = 2.0):
        """
        Initialize the persistent configuration manager.

        Args:
            app_name: Application name for configuration directory.
            config_filename: Name of the configuration file.
            debounce_seconds: Seconds to wait before writing after changes.
        """
        self.app_name = app_name
        self.config_filename = config_filename
        self.debounce_seconds = debounce_seconds

        # Determine config file path
        self.config_path = self._get_config_path()

        # Thread safety
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()

        # Debouncing
        self._pending_save = False
        self._save_timer: Optional[threading.Timer] = None

        # Configuration data
        self._config: Dict[str, Any] = {}

        print(f"[PersistentConfig] Config path: {self.config_path}")

    def _get_config_path(self) -> Path:
        """
        Get the platform-appropriate configuration file path.

        Returns:
            Path: Full path to the configuration file.
        """
        if os.name == 'nt':  # Windows
            base_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        elif os.name == 'posix':
            if 'darwin' in os.uname().sysname.lower():  # macOS
                base_dir = Path.home() / 'Library' / 'Application Support'
            else:  # Linux
                base_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        else:
            # Fallback
            base_dir = Path.home() / '.config'

        config_dir = base_dir / self.app_name
        return config_dir / self.config_filename

    def load(self, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from disk with fallback to defaults.

        Args:
            defaults: Default configuration values to use if file is missing/corrupted.

        Returns:
            Dict[str, Any]: Loaded configuration dictionary.
        """
        with self._lock:
            if defaults is None:
                defaults = {}

            # Try to load from file
            if self.config_path.exists():
                try:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)

                    # Merge with defaults (defaults for missing keys)
                    self._config = {**defaults, **loaded_config}
                    print(f"[PersistentConfig] Loaded configuration from {self.config_path}")
                    return self._config.copy()

                except (json.JSONDecodeError, IOError) as e:
                    print(f"[PersistentConfig] Error loading config: {e}")
                    print("[PersistentConfig] Using defaults due to corrupted config")
                    self._config = defaults.copy()
                    return self._config.copy()
            else:
                print(f"[PersistentConfig] Config file not found, using defaults")
                self._config = defaults.copy()
                return self._config.copy()

    def save(self, config: Dict[str, Any], debounce: bool = True):
        """
        Save configuration to disk with optional debouncing.

        Args:
            config: Configuration dictionary to save.
            debounce: If True, debounce writes. If False, write immediately.
        """
        with self._lock:
            self._config = config.copy()

        if debounce:
            self._schedule_save()
        else:
            self._write_to_disk()

    def update(self, **kwargs):
        """
        Update specific configuration values and trigger save.

        Args:
            **kwargs: Key-value pairs to update in configuration.
        """
        with self._lock:
            self._config.update(kwargs)

        self._schedule_save()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Any: Configuration value or default.
        """
        with self._lock:
            return self._config.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """
        Get a copy of all configuration values.

        Returns:
            Dict[str, Any]: Copy of all configuration.
        """
        with self._lock:
            return self._config.copy()

    def _schedule_save(self):
        """
        Schedule a debounced save operation.
        """
        with self._save_lock:
            # Cancel existing timer if present
            if self._save_timer is not None:
                self._save_timer.cancel()

            # Schedule new save
            self._save_timer = threading.Timer(
                self.debounce_seconds,
                self._write_to_disk
            )
            self._save_timer.daemon = True
            self._save_timer.start()
            self._pending_save = True

    def _write_to_disk(self):
        """
        Write configuration to disk (internal method).
        """
        with self._save_lock:
            self._pending_save = False
            self._save_timer = None

        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first (atomic write)
            temp_path = self.config_path.with_suffix('.tmp')

            with self._lock:
                config_to_save = self._config.copy()

            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)

            # Atomic replace (on most systems)
            temp_path.replace(self.config_path)

            print(f"[PersistentConfig] Configuration saved to {self.config_path}")

        except IOError as e:
            print(f"[PersistentConfig] Error saving configuration: {e}")

    def flush(self):
        """
        Force immediate write of pending changes to disk.
        """
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None

        if self._pending_save:
            self._write_to_disk()

    def reset_to_defaults(self, defaults: Dict[str, Any]):
        """
        Reset configuration to default values and save immediately.

        Args:
            defaults: Default configuration values.
        """
        with self._lock:
            self._config = defaults.copy()

        self._write_to_disk()
        print("[PersistentConfig] Configuration reset to defaults")


def test_persistent_config():
    """
    Test function for PersistentConfig.
    """
    print("Testing PersistentConfig...")

    # Create instance
    config = PersistentConfig(
        app_name="test_stickfigure",
        debounce_seconds=1.0
    )

    # Default values
    defaults = {
        'line_thickness': 4,
        'head_size': 0.4,
        'mouth_sensitivity': 0.025
    }

    # Load (will use defaults first time)
    loaded = config.load(defaults)
    print(f"Loaded config: {loaded}")

    # Update some values
    print("\nUpdating values...")
    config.update(line_thickness=6, head_size=0.5)

    # Wait for debounced save
    print("Waiting for debounced save (2 seconds)...")
    time.sleep(2.5)

    # Create new instance to test persistence
    print("\nCreating new instance to test persistence...")
    config2 = PersistentConfig(
        app_name="test_stickfigure",
        debounce_seconds=1.0
    )
    loaded2 = config2.load(defaults)
    print(f"Loaded config from disk: {loaded2}")

    # Verify values persisted
    assert loaded2['line_thickness'] == 6
    assert loaded2['head_size'] == 0.5
    assert loaded2['mouth_sensitivity'] == 0.025

    print("\n✓ All tests passed!")

    # Cleanup
    config2.config_path.unlink(missing_ok=True)
    config2.config_path.parent.rmdir()


if __name__ == "__main__":
    test_persistent_config()