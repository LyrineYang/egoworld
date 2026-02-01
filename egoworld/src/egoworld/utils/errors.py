"""Error classification for retry policy."""

from __future__ import annotations

from dataclasses import dataclass


class RetryableError(Exception):
    """Base class for retryable errors."""


class TransientIOError(RetryableError):
    pass


class OOMError(RetryableError):
    pass


class DecodeError(Exception):
    pass


class InvalidDataError(Exception):
    pass


class ModelMissingError(Exception):
    pass


@dataclass
class ErrorClassification:
    retryable: bool
    reason: str


def classify_error(error: Exception) -> ErrorClassification:
    if isinstance(error, RetryableError):
        return ErrorClassification(True, error.__class__.__name__)
    if isinstance(error, (DecodeError, InvalidDataError, ModelMissingError)):
        return ErrorClassification(False, error.__class__.__name__)
    message = str(error).lower()
    if "out of memory" in message:
        return ErrorClassification(True, "oom")
    if "cuda" in message and "error" in message:
        return ErrorClassification(True, "cuda_error")
    return ErrorClassification(False, "unknown")
