"""Transform for the llm-rubric assertion: what the grader is shown."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import checks  # noqa: E402


def rubric_view(output: Any, context: dict[str, Any]) -> str:
    return checks.rubric_view(checks.parse_response(output))
