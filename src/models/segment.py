"""Segment model for procedural boundaries in lab video."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """A time segment with a representative frame for task annotation."""

    segment_id: int
    start_time: float
    end_time: float
    representative_frame: str  # path to frame image

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
