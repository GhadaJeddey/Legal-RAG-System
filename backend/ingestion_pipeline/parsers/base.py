from abc import ABC, abstractmethod


class Parser(ABC):
    """Common interface for document parsing strategies."""

    @abstractmethod
    def extract(self, pdf_path: str, start_page: int | None = None, end_page: int | None = None) -> list[dict]:
        """Return a list of {"page": int, "markdown": str} entries."""
        raise NotImplementedError
