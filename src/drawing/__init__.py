#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł stick_figure - implementacja renderowania stick figure (patyczaka)
z płynną animacją ruchu rąk, nawet gdy nie są wykrywane przez kamerę.
"""

from .face_renderer import SimpleFaceRenderer
from .stick_figure_renderer import StickFigureRenderer

__all__ = ["StickFigureRenderer", "SimpleFaceRenderer"]
