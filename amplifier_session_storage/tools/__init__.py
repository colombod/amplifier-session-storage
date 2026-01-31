"""
Session tools for Amplifier agents.

This module provides tools that can be used by Amplifier agents (like session-analyst)
to interact with session storage without needing direct file access.
"""

from .session_tool import SessionTool, SessionToolConfig

__all__ = [
    "SessionTool",
    "SessionToolConfig",
]
