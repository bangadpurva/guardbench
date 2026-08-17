"""No-guard baseline. Never blocks; establishes the undefended ASR ceiling."""
from __future__ import annotations

from .base import GuardrailBase


class G0NoGuard(GuardrailBase):
    name = "none"

    def _evaluate(self, prompt: str):
        return False, 0.0, {}
