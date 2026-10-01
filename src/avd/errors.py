"""Expected, actionable pipeline failures."""


class DubbingError(RuntimeError):
    """An operation could not safely produce a complete dub."""
