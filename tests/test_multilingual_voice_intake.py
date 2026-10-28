"""
Unit and Integration Tests for Multilingual Voice-to-Text Complaint Intake.
Validates English, Hindi, and Marathi speech/text processing, safety overrides,
linguistic normalization, and complaint pipeline persistence.
"""

import pytest
from grievanceroute.services.multilingual_service import (
    detect_language,
    normalize_multilingual_text,
    check_multilingual_electrical_hazard,
)
from grievanceroute.services.prediction_service import PredictionService
from grievanceroute.services.complaint_service import ComplaintService
from grievanceroute.database.repository import GrievanceRepository


def test_language_detection():
    """Verify correct script and lexical marker detection across languages."""
    # English
    assert detect_language("Deep pothole on the road near Balgandharva chowk") == "English / Romanized"
    
    # Marathi
    marathi_sample = "रस्त्यावर खूप मोठा खड्डा पडला आहे आणि वाहने घसरून अपघात होत आहेत"
    assert detect_language(marathi_sample) == "Marathi"

    # Hindi
    hindi_sample = "सड़क पर बहुत कचरा फैला हुआ है और बदबू आ रही है"
    assert detect_language(hindi_sample) == "Hindi"


def test_multilingual_electrical_hazard_safety_check():
    """Verify electrical hazard terms in English, Hindi, and Marathi trigger safety warnings."""
    # English
    has_hazard, msg = check_multilingual_electrical_hazard("Live electrical wire sparking violently near road")
    assert has_hazard is True
    assert "Electrical hazard term" in msg

    # Marathi
    has_hazard_mr, msg_mr = check_multilingual_electrical_hazard("विजेची जिवंत तार रस्त्यावर पडली आहे आणि ठिणग्या उडत आहेत")
    assert has_hazard_mr is True
    assert "विजेची तार" in msg_mr or "ठिणग्या" in msg_mr

    # Hindi
    has_hazard_hi, msg_hi = check_multilingual_electrical_hazard("सड़क पर बिजली का तार टूट गया है और चिंगारी निकल रही है")
    assert has_hazard_hi is True
    assert "बिजली का तार" in msg_hi or "चिंगारी" in msg_hi

    # Normal non-hazardous complaint
    no_hazard, no_msg = check_multilingual_electrical_hazard("Normal garbage accumulation near school")
    assert no_hazard is False
    assert no_msg is None


def test_prediction_service_marathi_pothole():
    """Verify Marathi complaint predicts pothole with elevated priority."""
    pred_svc = PredictionService()
    text = "रस्त्यावर खूप मोठा खड्डा पडला आहे आणि अपघात होत आहेत"
    result = pred_svc.predict_complaint(text)

    assert result["predicted_category"] == "pothole"
    assert result["confidence"] > 0.70
    assert result["detected_language"] == "Marathi"
    assert result["crew_required"] == "Road Maintenance"
    assert result["urgency_score"] >= 3


def test_prediction_service_marathi_electrical_safety_override():
    """Verify Marathi electrical hazard triggers Category=streetlight, Urgency=5, and Safety Override."""
    pred_svc = PredictionService()
    text = "विजेची जिवंत तार रस्त्यावर तुटून पडली आहे आणि ठिणग्या उडत आहेत"
    result = pred_svc.predict_complaint(text)

    assert result["safety_override"] is True
    assert result["predicted_category"] == "streetlight"
    assert result["urgency_score"] == 5
    assert result["crew_required"] == "Electrical Crew"
    assert "Safety Escalation" in result["safety_warning"]


def test_prediction_service_hindi_water_leak():
    """Verify Hindi pipeline leak predicts water leak with Water Supply Crew."""
    pred_svc = PredictionService()
    text = "मुख्य पीने के पानी की पाइपलाइन फट गई है और सड़क पर पानी बह रहा है"
    result = pred_svc.predict_complaint(text)

    assert result["predicted_category"] == "water leak"
    assert result["detected_language"] == "Hindi"
    assert result["crew_required"] == "Water Supply Crew"


def test_prediction_service_hindi_garbage():
    """Verify Hindi garbage complaint predicts sanitation crew."""
    pred_svc = PredictionService()
    text = "सड़क पर कचरा फैला हुआ है और बदबू आ रही है"
    result = pred_svc.predict_complaint(text)

    assert result["predicted_category"] == "garbage"
    assert result["crew_required"] == "Sanitation Crew"
    assert result["detected_language"] == "Hindi"


def test_complaint_service_marathi_submission_end_to_end():
    """Verify submitting a Marathi complaint persists in DB and returns triage."""
    repo = GrievanceRepository()
    svc = ComplaintService(repo=repo)

    marathi_complaint = "रस्त्यावर मोठा खड्डा पडला असून दुचाकीस्वार पडत आहेत"
    success, msg, record = svc.submit_complaint(
        complaint_text=marathi_complaint,
        latitude=18.5074,
        longitude=73.8077,
        zone="Kothrud",
    )

    assert success is True
    assert "triaged and persisted" in msg
    assert record["complaint_text"] == marathi_complaint
    assert record["complaint_type"] == "pothole"
    assert record["crew_required"] == "Road Maintenance"
    assert record["detected_language"] == "Marathi"
