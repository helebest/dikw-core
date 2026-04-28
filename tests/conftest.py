"""Pytest session bootstrap.

Windows ships ``asyncio.ProactorEventLoop`` as the default, which psycopg's
async client refuses to run under (raises ``InterfaceError`` at connect).
Switch the policy to ``WindowsSelectorEventLoopPolicy`` so the Postgres
storage-contract tests run locally; Linux / macOS / CI are unaffected.
"""

from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
