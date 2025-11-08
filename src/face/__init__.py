#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/face/__init__.py
"""
Moduł do wykrywania i analizy twarzy użytkownika przy pomocy MediaPipe FaceMesh.
"""

from src.face.face_mesh_detector import FaceMeshDetector

__all__ = ["FaceMeshDetector"]
