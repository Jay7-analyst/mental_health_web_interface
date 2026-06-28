from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredictionResult:
    detected_language: str
    selected_model: str
    predicted_category: str
    confidence_score: Optional[float] = None
    class_probabilities: dict[str, float] = field(default_factory=dict)
    mode: str = "demo"
    confidence_available: bool = True
    decision_strength: Optional[str] = None
    note: Optional[str] = None

    def confidence_percent(self) -> Optional[int]:
        if self.confidence_score is None:
            return None
        return int(round(self.confidence_score * 100))
