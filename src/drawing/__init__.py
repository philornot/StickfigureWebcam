#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# drawing/__init__.py

"""
Moduł stick_figure - prosta implementacja renderowania stick figure (patyczaka)
skupiona tylko na popiersu, z prostą mimiką twarzy.
"""

from .face_renderer import SimpleFaceRenderer
from .stick_figure_renderer import StickFigureRenderer

__all__ = ['StickFigureRenderer', 'SimpleFaceRenderer']
