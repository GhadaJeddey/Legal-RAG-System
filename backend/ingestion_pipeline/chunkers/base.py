from abc import ABC, abstractmethod


class Chunker(ABC):
    """Common interface for chunking strategies."""

    @abstractmethod
    def chunk(self, markdown: str) -> list[dict]:
        """Return a list of chunk dicts, each with at least a 'text' and 'metadata' key."""
        raise NotImplementedError
