#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for performance monitor (performance.py).
"""

import unittest
from unittest.mock import patch

from src.utils.performance import PerformanceMonitor


class TestPerformanceMonitor(unittest.TestCase):
    """
    Tests for PerformanceMonitor class that measures execution time, FPS
    and stores measurement history.
    """

    def setUp(self):
        """Initialize before each test."""
        # Initialize performance monitor
        self.monitor = PerformanceMonitor("TestModule", history_size=5)

    def test_initialization(self):
        """Test performance monitor initialization."""
        # Check if initialization completed correctly
        self.assertEqual(self.monitor.module_name, "TestModule")
        self.assertEqual(self.monitor.history_size, 5)
        self.assertIsNone(self.monitor.start_time)
        self.assertEqual(len(self.monitor.execution_times), 0)
        self.assertEqual(len(self.monitor.fps_history), 0)
        self.assertEqual(self.monitor.frame_counter, 0)
        self.assertEqual(len(self.monitor.timing_markers), 0)
        self.assertEqual(len(self.monitor.segment_times), 0)

    @patch("time.time")
    def test_start_timer(self, mock_time):
        """Test starting time measurement."""
        # Configure time.time mock
        mock_time.return_value = 10.0

        # Start measurement
        self.monitor.start_timer()

        # Check if start time was saved
        self.assertEqual(self.monitor.start_time, 10.0)

    @patch("time.time")
    def test_stop_timer(self, mock_time):
        """Test stopping time measurement."""
        # Configure time.time mock for different calls
        mock_time.side_effect = [10.0, 10.5, 11.0]

        # Start measurement
        self.monitor.start_timer()

        # Stop measurement
        execution_time = self.monitor.stop_timer()

        # Check results
        self.assertEqual(execution_time, 0.5)  # 10.5 - 10.0
        self.assertEqual(len(self.monitor.execution_times), 1)
        self.assertEqual(self.monitor.execution_times[0], 0.5)
        self.assertEqual(self.monitor.frame_counter, 1)

        # Don't test FPS update, because it requires more mock preparation
        # and is more prone to errors

    def test_get_last_execution_time(self):
        """Test getting last execution time."""
        # When no measurements
        self.assertEqual(self.monitor.get_last_execution_time(), 0.0)

        # Add measurement
        self.monitor.execution_times.append(0.5)

        # Check last time
        self.assertEqual(self.monitor.get_last_execution_time(), 0.5)

    def test_get_average_execution_time(self):
        """Test calculating average execution time."""
        # When no measurements
        self.assertEqual(self.monitor.get_average_execution_time(), 0.0)

        # Add measurements
        self.monitor.execution_times.extend([0.1, 0.2, 0.3, 0.4, 0.5])

        # Check average of all measurements
        self.assertAlmostEqual(self.monitor.get_average_execution_time(), 0.3, places=6)

        # Check average of last 3 measurements
        self.assertAlmostEqual(
            self.monitor.get_average_execution_time(num_samples=3), 0.4, places=6
        )

    def test_get_current_fps(self):
        """Test getting current FPS."""
        # When no measurements
        self.assertEqual(self.monitor.get_current_fps(), 0.0)

        # Add FPS measurement
        self.monitor.fps_history.append(30.0)

        # Check current FPS
        self.assertEqual(self.monitor.get_current_fps(), 30.0)

    def test_get_average_fps(self):
        """Test calculating average FPS."""
        # When no measurements
        self.assertEqual(self.monitor.get_average_fps(), 0.0)

        # Add FPS measurements
        self.monitor.fps_history.extend([25.0, 27.0, 30.0, 28.0, 26.0])

        # Check average of all measurements
        self.assertEqual(self.monitor.get_average_fps(), 27.2)

        # Check average of last 3 measurements
        self.assertEqual(self.monitor.get_average_fps(num_samples=3), 28.0)

    @patch("time.time")
    def test_mark_time(self, mock_time):
        """Test saving time markers."""
        # Configure time.time mock
        mock_time.return_value = 10.0

        # Save time marker
        self.monitor.mark_time("start")

        # Check if marker was saved
        self.assertEqual(len(self.monitor.timing_markers), 1)
        self.assertEqual(self.monitor.timing_markers["start"], 10.0)

    @patch("time.time")
    def test_measure_segment(self, mock_time):
        """Test measuring time between markers."""
        # Configure time.time mock for different calls
        mock_time.side_effect = [10.0, 10.5]

        # Save time markers
        self.monitor.mark_time("start")
        self.monitor.mark_time("end")

        # Measure segment
        duration = self.monitor.measure_segment("start", "end", "test_segment")

        # Check results
        self.assertEqual(duration, 0.5)
        self.assertEqual(len(self.monitor.segment_times), 1)
        self.assertEqual(len(self.monitor.segment_times["test_segment"]), 1)
        self.assertEqual(self.monitor.segment_times["test_segment"][0], 0.5)

        # Test for non-existent markers
        self.assertEqual(self.monitor.measure_segment("non_existent", "end", "invalid"), -1.0)

    def test_get_segment_stats(self):
        """Test getting segment statistics."""
        # When no segment
        self.assertEqual(self.monitor.get_segment_stats("non_existent"), (0.0, 0.0, 0.0))

        # Add measurements for segment
        self.monitor.segment_times["test_segment"] = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Check statistics
        avg, min_time, max_time = self.monitor.get_segment_stats("test_segment")
        self.assertAlmostEqual(avg, 0.3, places=6)
        self.assertEqual(min_time, 0.1)
        self.assertEqual(max_time, 0.5)

    def test_reset(self):
        """Test resetting performance monitor."""
        # Add data
        self.monitor.start_time = 10.0
        self.monitor.execution_times.append(0.5)
        self.monitor.fps_history.append(30.0)
        self.monitor.frame_counter = 10
        self.monitor.timing_markers["start"] = 10.0
        self.monitor.segment_times["test_segment"] = [0.5]

        # Reset
        self.monitor.reset()

        # Check if everything was reset
        self.assertIsNone(self.monitor.start_time)
        self.assertEqual(len(self.monitor.execution_times), 0)
        self.assertEqual(len(self.monitor.fps_history), 0)
        self.assertEqual(self.monitor.frame_counter, 0)
        self.assertEqual(len(self.monitor.timing_markers), 0)
        self.assertEqual(len(self.monitor.segment_times), 0)

    def test_get_performance_summary(self):
        """Test creating performance summary."""
        # Add data
        self.monitor.execution_times.extend([0.1, 0.2, 0.3, 0.4, 0.5])
        self.monitor.fps_history.extend([25.0, 27.0, 30.0, 28.0, 26.0])

        # Add segment
        self.monitor.segment_times["test_segment"] = [0.1, 0.2, 0.3]

        # Get summary
        summary = self.monitor.get_performance_summary()

        # Check summary content
        self.assertEqual(summary["module"], "TestModule")
        self.assertAlmostEqual(summary["avg_execution_time"], 0.3, places=6)
        self.assertAlmostEqual(summary["avg_execution_time_ms"], 300.0, places=6)
        self.assertEqual(summary["last_execution_time"], 0.5)
        self.assertEqual(summary["avg_fps"], 27.2)
        self.assertEqual(summary["current_fps"], 26.0)
        self.assertEqual(summary["samples_count"], 5)

        # Check segment statistics
        self.assertIn("segments", summary)
        self.assertIn("test_segment", summary["segments"])
        segment_stats = summary["segments"]["test_segment"]
        self.assertAlmostEqual(segment_stats["avg"], 0.2, places=6)
        self.assertEqual(segment_stats["min"], 0.1)
        self.assertEqual(segment_stats["max"], 0.3)


if __name__ == "__main__":
    unittest.main()
