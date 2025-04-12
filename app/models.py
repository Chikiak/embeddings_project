# app/models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)  # Use frozen=True for immutable result objects
class SearchResultItem:
    """
    Represents a single item found during a similarity search.
    """
    id: str  # Unique identifier, typically the image file path
    distance: Optional[float] = None  # Distance score from the query (lower is more similar for cosine)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Associated metadata

    @property
    def similarity(self) -> Optional[float]:
        """
        Calculates similarity (assuming distance is cosine distance).
        Similarity = 1 - distance. Returns None if distance is None.
        """
        if self.distance is None:
            return None
        # Clamp between 0 and 1, although cosine distance should be >= 0
        return max(0.0, 1.0 - self.distance)


@dataclass(frozen=True)
class SearchResults:
    """
    Represents the complete results of a similarity search query.
    """
    items: List[SearchResultItem] = field(default_factory=list)
    query_vector: Optional[List[float]] = None  # Optional: include the vector used for the query

    @property
    def count(self) -> int:
        """Returns the number of result items."""
        return len(self.items)

    @property
    def is_empty(self) -> bool:
        """Checks if the result set contains no items."""
        return not self.items
