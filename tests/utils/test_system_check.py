#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testy jednostkowe dla modułu sprawdzania wymagań systemowych (system_check.py).
"""

import os
import platform
import unittest
from unittest.mock import MagicMock, patch, call

from src.utils.system_check import SystemCheck, check_system_requirements


class TestSystemCheck(unittest.TestCase):
    """
    Testy dla klasy SystemCheck, która sprawdza dostępność wymaganych komponentów.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Mock loggera
        self.mock_logger = MagicMock()

        # Patchujemy wykrywanie systemu, aby zwracało zawsze 'Windows'
        self.platform_system_patch = patch('platform.system', return_value='Windows')
        self.platform_system = self.platform_system_patch.start()

        # Patchujemy import pyvirtualcam
        self.pyvirtualcam_patch = patch('src.utils.system_check.PYVIRTUALCAM_AVAILABLE', True)
        self.pyvirtualcam_available = self.pyvirtualcam_patch.start()

        # Patchujemy import mediapipe
        self.mediapipe_patch = patch('src.utils.system_check.MEDIAPIPE_AVAILABLE', True)
        self.mediapipe_available = self.mediapipe_patch.start()

        # Patchujemy cv2.VideoCapture
        self.cv2_videocapture_patch = patch('cv2.VideoCapture')
        self.mock_cv2_videocapture = self.cv2_videocapture_patch.start()

        # Mock dla obiektu kamery
        self.mock_camera = MagicMock()
        self.mock_camera.isOpened.return_value = True
        self.mock_camera.read.return_value = (True, MagicMock())
        self.mock_camera.get.side_effect = lambda prop: {
            0: 640,  # CAP_PROP_FRAME_WIDTH
            1: 480,  # CAP_PROP_FRAME_HEIGHT
            5: 30  # CAP_PROP_FPS
        }.get(prop, 0)

        self.mock_cv2_videocapture.return_value = self.mock_camera

        # Inicjalizacja SystemCheck
        self.system_check = SystemCheck(logger=self.mock_logger)

    def tearDown(self):
        """Sprzątanie po każdym teście."""
        self.platform_system_patch.stop()
        self.pyvirtualcam_patch.stop()
        self.mediapipe_patch.stop()
        self.cv2_videocapture_patch.stop()

    def test_initialization(self):
        """Test inicjalizacji obiektu SystemCheck."""
        # Sprawdzamy czy inicjalizacja przebiegła poprawnie
        self.assertEqual(self.system_check.system, 'Windows')
        self.assertIsNotNone(self.system_check.results)
        self.assertIsNotNone(self.system_check.install_links)

        # Sprawdzamy czy wszystkie wymagane komponenty są w wynikach
        for component in ['camera', 'virtual_camera', 'mediapipe', 'obs', 'v4l2loopback']:
            self.assertIn(component, self.system_check.results)

    def test_check_camera_success(self):
        """Test sprawdzania kamery gdy kamera jest dostępna."""
        # Sprawdzamy kamerę
        result = self.system_check.check_camera(camera_id=0)

        # Sprawdzamy wyniki
        self.assertTrue(result['status'])
        self.assertIn('Kamera o ID: 0 działa poprawnie', result['message'])
        self.assertEqual(result['details']['camera_id'], 0)
        self.assertEqual(result['details']['width'], 640)
        self.assertEqual(result['details']['height'], 480)
        self.assertEqual(result['details']['fps'], 30)

    def test_check_camera_failure(self):
        """Test sprawdzania kamery gdy kamera nie jest dostępna."""
        # Zmieniamy mock kamery, aby symulować niedostępność
        self.mock_camera.isOpened.return_value = False

        # Sprawdzamy kamerę
        result = self.system_check.check_camera(camera_id=0)

        # Sprawdzamy wyniki
        self.assertFalse(result['status'])
        self.assertIn('Nie można otworzyć kamery', result['message'])

    @patch('pyvirtualcam.Camera')
    def test_check_virtual_camera_success(self, mock_pyvirtualcam_camera):
        """Test sprawdzania wirtualnej kamery gdy jest dostępna."""
        # Konfigurujemy mocka pyvirtualcam.Camera
        mock_camera = MagicMock()
        mock_camera.backend = 'obs'
        mock_camera.width = 320
        mock_camera.height = 240
        mock_camera.fps = 20
        mock_pyvirtualcam_camera.return_value = mock_camera

        # Sprawdzamy wirtualną kamerę
        result = self.system_check.check_virtual_camera()

        # Sprawdzamy wyniki
        self.assertTrue(result['status'])
        self.assertIn('Wirtualna kamera działa poprawnie', result['message'])
        self.assertEqual(result['details']['backend'], 'obs')
        self.assertEqual(result['details']['width'], 320)
        self.assertEqual(result['details']['height'], 240)
        self.assertEqual(result['details']['fps'], 20)

    @patch('pyvirtualcam.Camera')
    def test_check_virtual_camera_failure(self, mock_pyvirtualcam_camera):
        """Test sprawdzania wirtualnej kamery gdy nie jest dostępna."""
        # Konfigurujemy mocka pyvirtualcam.Camera, aby rzucał wyjątek
        mock_pyvirtualcam_camera.side_effect = Exception("Virtual camera not available")

        # Sprawdzamy wirtualną kamerę
        result = self.system_check.check_virtual_camera()

        # Sprawdzamy wyniki
        self.assertFalse(result['status'])
        self.assertIn('Błąd podczas sprawdzania wirtualnej kamery', result['message'])
        self.assertEqual(result['details']['error'], 'Virtual camera not available')

    @patch('mediapipe.solutions.pose')
    def test_check_mediapipe_success(self, mock_mp_pose):
        """Test sprawdzania MediaPipe gdy jest dostępny."""
        # Konfigurujemy mocka dla MediaPipe
        mock_pose = MagicMock()
        mock_mp_pose.Pose.return_value = mock_pose

        # Dodajemy mock dla wersji MediaPipe
        with patch('mediapipe.__version__', '0.8.10'):
            # Sprawdzamy MediaPipe
            result = self.system_check.check_mediapipe()

            # Sprawdzamy wyniki
            self.assertTrue(result['status'])
            self.assertIn('MediaPipe działa poprawnie', result['message'])
            self.assertEqual(result['details']['version'], '0.8.10')

    def test_check_mediapipe_not_installed(self):
        """Test sprawdzania MediaPipe gdy nie jest zainstalowany."""
        # Patchujemy MEDIAPIPE_AVAILABLE na False
        with patch('src.utils.system_check.MEDIAPIPE_AVAILABLE', False):
            # Sprawdzamy MediaPipe
            result = self.system_check.check_mediapipe()

            # Sprawdzamy wyniki
            self.assertFalse(result['status'])
            self.assertIn('MediaPipe nie jest zainstalowana', result['message'])
            self.assertEqual(result['details']['install_command'], 'pip install mediapipe')

    @patch('os.path.exists')
    def test_check_obs_installed(self, mock_exists):
        """Test sprawdzania OBS gdy jest zainstalowany."""
        # Konfigurujemy mocka dla os.path.exists
        mock_exists.return_value = True

        # Sprawdzamy OBS
        result = self.system_check.check_obs()

        # Sprawdzamy wyniki
        self.assertTrue(result['status'])
        self.assertIn('OBS Studio jest zainstalowany', result['message'])

    @patch('os.path.exists')
    def test_check_obs_not_installed(self, mock_exists):
        """Test sprawdzania OBS gdy nie jest zainstalowany."""
        # Konfigurujemy mocka dla os.path.exists
        mock_exists.return_value = False

        # Sprawdzamy OBS
        result = self.system_check.check_obs()

        # Sprawdzamy wyniki
        self.assertFalse(result['status'])
        self.assertIn('OBS Studio nie jest zainstalowany', result['message'])

    def test_get_missing_components_none_missing(self):
        """Test pobierania brakujących komponentów gdy wszystkie są dostępne."""
        # Ustawiamy wszystkie komponenty jako dostępne
        for component in self.system_check.results:
            self.system_check.results[component]['status'] = True

        # Pobieramy brakujące komponenty
        missing = self.system_check.get_missing_components()

        # Sprawdzamy czy lista jest pusta
        self.assertEqual(len(missing), 0)

    def test_get_missing_components_some_missing(self):
        """Test pobierania brakujących komponentów gdy niektóre są niedostępne."""
        # Najpierw resetujemy wszystkie do True
        for component in self.system_check.results:
            self.system_check.results[component]['status'] = True

        # Teraz ustawiamy kilka jako niedostępne
        self.system_check.results['virtual_camera']['status'] = False
        self.system_check.results['obs']['status'] = False

        # Pobieramy brakujące komponenty
        missing = self.system_check.get_missing_components()

        # Sprawdzamy liczbę brakujących komponentów
        self.assertEqual(len(missing), 2)

        # Sprawdzamy czy odpowiednie komponenty są na liście
        missing_components = [item['name'] for item in missing]
        self.assertIn('virtual_camera', missing_components)
        self.assertIn('obs', missing_components)

    def test_are_all_requirements_met_all_met(self):
        """Test sprawdzania czy wszystkie wymagania są spełnione gdy są."""
        # Ustawiamy wszystkie komponenty jako dostępne
        self.system_check.results['camera']['status'] = True
        self.system_check.results['virtual_camera']['status'] = True
        self.system_check.results['mediapipe']['status'] = True
        self.system_check.results['obs']['status'] = True

        # Sprawdzamy czy wszystkie wymagania są spełnione
        result = self.system_check.are_all_requirements_met()

        # Sprawdzamy wynik
        self.assertTrue(result)

    def test_are_all_requirements_met_some_not_met(self):
        """Test sprawdzania czy wszystkie wymagania są spełnione gdy niektóre nie są."""
        # Ustawiamy niektóre komponenty jako niedostępne
        self.system_check.results['camera']['status'] = True
        self.system_check.results['virtual_camera']['status'] = False
        self.system_check.results['mediapipe']['status'] = True
        self.system_check.results['obs']['status'] = True

        # Sprawdzamy czy wszystkie wymagania są spełnione
        result = self.system_check.are_all_requirements_met()

        # Sprawdzamy wynik
        self.assertFalse(result)

    def test_get_installation_instructions(self):
        """Test pobierania instrukcji instalacji brakujących komponentów."""
        # Ustawiamy niektóre komponenty jako niedostępne
        self.system_check.results['camera']['status'] = True
        self.system_check.results['virtual_camera']['status'] = False
        self.system_check.results['mediapipe']['status'] = False
        self.system_check.results['obs']['status'] = True

        # Pobieramy instrukcje instalacji
        instructions = self.system_check.get_installation_instructions()

        # Sprawdzamy czy instrukcje zawierają odpowiednie komponenty
        self.assertIn('virtual_camera', instructions)
        self.assertIn('mediapipe', instructions)
        self.assertNotIn('camera', instructions)
        self.assertNotIn('obs', instructions)

        # Sprawdzamy czy instrukcje dla virtual_camera zawierają OBS (dla Windows)
        self.assertTrue(any('OBS' in instr for instr in instructions['virtual_camera']))

    @patch('src.utils.system_check.SystemCheck')
    def test_check_system_requirements(self, mock_system_check_class):
        """Test funkcji check_system_requirements."""
        # Konfigurujemy mocka dla SystemCheck
        mock_system_check = MagicMock()
        mock_system_check.check_all.return_value = {'some': 'results'}
        mock_system_check.are_all_requirements_met.return_value = True
        mock_system_check.get_missing_components.return_value = []
        mock_system_check.get_installation_instructions.return_value = {}

        mock_system_check_class.return_value = mock_system_check

        # Wywołujemy funkcję
        all_met, results = check_system_requirements(logger=self.mock_logger)

        # Sprawdzamy wyniki
        self.assertTrue(all_met)
        self.assertEqual(results['all_met'], True)
        self.assertEqual(results['missing'], [])
        self.assertEqual(results['instructions'], {})

        # Sprawdzamy czy metody SystemCheck zostały wywołane
        mock_system_check.check_all.assert_called_once()
        mock_system_check.are_all_requirements_met.assert_called_once()
        mock_system_check.get_missing_components.assert_called_once()
        mock_system_check.get_installation_instructions.assert_called_once()


if __name__ == "__main__":
    unittest.main()
