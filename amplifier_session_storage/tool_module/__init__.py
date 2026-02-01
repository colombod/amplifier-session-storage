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

from .tool import SessionToolModule, ToolResult, create_tool, mount

__all__ = [
    "SessionToolModule",
    "ToolResult",
    "create_tool",
    "mount",
]
