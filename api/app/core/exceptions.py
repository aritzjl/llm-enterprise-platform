"""Application-level exception types."""


class UpstreamServiceError(Exception):
    """Raised when the configured LLM provider cannot satisfy a request."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)
