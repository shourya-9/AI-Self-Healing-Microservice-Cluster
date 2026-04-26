"""
Shared pytest configuration and fixtures.

docker is mocked at the sys.modules level here so that importing
ai-orchestrator/main.py (which calls docker.from_env() at module load
time) never requires a live Docker daemon during tests.
"""
import sys
from unittest.mock import MagicMock

# Must happen before any test file imports the orchestrator module.
sys.modules.setdefault("docker", MagicMock())
