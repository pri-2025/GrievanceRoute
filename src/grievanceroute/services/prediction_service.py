"""
Prediction Service for GrievanceRoute.
Coordinates ML inference pipelines, category prediction with confidence probabilities,
urgency estimation, resolution-time forecasting, and safety-critical rule overrides.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from grievanceroute.ml.classifier import load_classifier_pipeline
from grievanceroute.ml.regression import load_regressor_pipeline
from grievanceroute.services.multilingual_service import (
    normalize_multilingual_text,
    MULTILINGUAL_URGENCY_KEYWORDS,
)


ELECTRICAL_HAZARD_KEYWORDS = [
    "live electrical wire",
    "live wire",
    "exposed wire",
    "exposed electrical wire",
    "sparking",
    "transformer blast",
    "electric shock",
    "electrocution",
]

URGENCY_KEYWORD_RULES = {
    5: ["electrocution", "transformer blast", "severe flooding", "immediate danger", "life threatening"],
    4: ["deep pothole", "burst", "burst pipe", "spark", "accident", "hospital", "dead animal"],
    3: ["for three days", "for 3 days", "repeatedly", "overflowing", "no water"],
}

CATEGORY_TO_CREW = {
    "pothole": "Road Maintenance",
    "water leak": "Water Supply Crew",
    "streetlight": "Electrical Crew",
    "garbage": "Sanitation Crew",
}


class PredictionService:
    """Provides inference for complaint triage."""

    def __init__(
        self,
        classifier_path: str | Path = "artifacts/models/complaint_classifier.pkl",
        urgency_path: str | Path = "artifacts/models/urgency_regressor.pkl",
        resolution_path: str | Path = "artifacts/models/resolution_regressor.pkl",
    ):
        self.classifier_path = Path(classifier_path)
        self.urgency_path = Path(urgency_path)
        self.resolution_path = Path(resolution_path)

        self.classifier: Optional[Pipeline] = None
        self.urgency_regressor: Optional[Pipeline] = None
        self.resolution_regressor: Optional[Pipeline] = None

        self._load_models_if_available()

    def _load_models_if_available(self) -> None:
        if self.classifier_path.exists():
            try:
                self.classifier = load_classifier_pipeline(self.classifier_path)
            except Exception:
                self.classifier = None

        if self.urgency_path.exists():
            try:
                self.urgency_regressor = load_regressor_pipeline(self.urgency_path)
            except Exception:
                self.urgency_regressor = None

        if self.resolution_path.exists():
            try:
                self.resolution_regressor = load_regressor_pipeline(self.resolution_path)
            except Exception:
                self.resolution_regressor = None

    def predict_complaint(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run end-to-end triage prediction on text and context.
        Applies multilingual normalization, ML inference, and safety-rule escalation.
        """
        clean_text = text.strip()
        lower_text = clean_text.lower()

        # Multilingual normalization (English, Hindi, Marathi)
        multi_info = normalize_multilingual_text(clean_text)
        triage_text = multi_info["triage_text"]
        detected_language = multi_info["detected_language"]

        # 1. Base ML Category Prediction
        predicted_category = "pothole"
        confidence = 0.50
        probabilities: Dict[str, float] = {}

        if self.classifier is not None:
            # Use triage_text for classifier inference
            preds = self.classifier.predict([triage_text])
            predicted_category = preds[0]
            if hasattr(self.classifier, "predict_proba"):
                probs = self.classifier.predict_proba([triage_text])[0]
                classes = self.classifier.classes_
                probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
                confidence = float(np.max(probs))

        # 2. Safety Rule Check (Electrical Hazards in English, Hindi, and Marathi)
        override_applied = False
        warning_msg = None

        # Check English keyword list
        for hazard in ELECTRICAL_HAZARD_KEYWORDS:
            if hazard in lower_text:
                override_applied = True
                predicted_category = "streetlight"
                warning_msg = f"Safety Escalation: '{hazard}' detected. Prioritized for immediate electrical response."
                break

        # Check Multilingual electrical hazard
        if not override_applied and multi_info.get("is_electrical_hazard"):
            override_applied = True
            predicted_category = "streetlight"
            warning_msg = multi_info.get("hazard_warning", "Safety Escalation: Electrical hazard detected.")

        # 3. Urgency Estimation
        urgency_score = 3
        if override_applied or any(h in lower_text for h in ELECTRICAL_HAZARD_KEYWORDS):
            urgency_score = 5
        elif self.urgency_regressor is not None and context is not None:
            # Build DataFrame row matching schema
            df_ctx = pd.DataFrame([context])
            u_pred = self.urgency_regressor.predict(df_ctx)[0]
            urgency_score = int(np.clip(round(u_pred), 1, 5))
        else:
            # Keyword fallback (English & Multilingual)
            for score, words in URGENCY_KEYWORD_RULES.items():
                if any(w in lower_text for w in words):
                    urgency_score = max(urgency_score, score)
            for score, words in MULTILINGUAL_URGENCY_KEYWORDS.items():
                if any(w in clean_text or w in triage_text for w in words):
                    urgency_score = max(urgency_score, score)

        # 4. Resolution Time Estimation (Hours)
        est_resolution_hours = 24.0
        if self.resolution_regressor is not None and context is not None:
            df_ctx = pd.DataFrame([context])
            r_pred = self.resolution_regressor.predict(df_ctx)[0]
            est_resolution_hours = round(float(np.clip(r_pred, 1.0, 168.0)), 1)
        else:
            base_times = {1: 48.0, 2: 36.0, 3: 24.0, 4: 16.0, 5: 8.0}
            est_resolution_hours = base_times.get(urgency_score, 24.0)

        crew_required = CATEGORY_TO_CREW.get(predicted_category, "Road Maintenance")

        return {
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "safety_override": override_applied,
            "safety_warning": warning_msg,
            "urgency_score": urgency_score,
            "estimated_resolution_hours": est_resolution_hours,
            "crew_required": crew_required,
            "detected_language": detected_language,
            "triage_text": triage_text,
        }

