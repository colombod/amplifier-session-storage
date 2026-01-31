"""
Amplifier tool module for session operations.

This module provides the `session` tool that agents can use for safe,
structured access to session data.

Usage in Amplifier behaviors:

```yaml
tools:
  - module: tool-session
    source: amplifier-session-storage
```
"""

from .tool import SessionToolModule, create_tool

__all__ = [
    "SessionToolModule",
    "create_tool",
]
