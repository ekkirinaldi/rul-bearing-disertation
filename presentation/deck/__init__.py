"""Dissertation presentation builder.

Turns a YAML deck specification into a .pptx that matches the approved
ITB doctoral-defence template.
"""

from .builder import build_deck, load_spec, lint_spec

__all__ = ["build_deck", "load_spec", "lint_spec"]
__version__ = "1.0.0"
