"""
Multilingual Text Processing and Normalization Service for GrievanceRoute.
Supports English, Hindi, and Marathi citizen complaint processing,
language detection, phonetic transliteration (via AnyAscii), domain-specific
civic keyword mapping, and safety escalation keyword detection.
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple
import anyascii


# Marathi & Hindi Electrical Hazard Terms (Devanagari and Romanized)
MULTILINGUAL_ELECTRICAL_HAZARDS = [
    # Marathi Devanagari
    "विजेची तार", "विद्युत तार", "उघडी तार", "नंगी तार", "ठिणग्या", "विद्युत धक्का", 
    "करंट लागतो", "करंट लागला", "शॉर्ट सर्किट", "लाईव्ह वायर", "लाईव वायर", 
    "ट्रान्सफॉर्मर ब्लास्ट", "विजेचा धक्का",
    # Hindi Devanagari
    "बिजली का तार", "बिजली की तार", "बिजली का झटका", "नंगा तार", "चिंगारी", 
    "बिजली का करंट", "करंट आ रहा", "ट्रांसफार्मर ब्लास्ट", "बिजली का खंभा स्पार्क",
    # Phonetic / Romanized / Keywords
    "live wire", "exposed wire", "sparking", "electric shock", "electrocution",
    "transformer blast", "short circuit", "current lagto", "bijli ka tar",
    "vijechi tar", "vidyut tar", "chingari", "thingya"
]

# High Urgency Keywords (Devanagari + English)
MULTILINGUAL_URGENCY_KEYWORDS = {
    5: [
        "electrocution", "transformer blast", "severe flooding", "immediate danger", "life threatening",
        "विद्युत धक्का", "बिजली का झटका", "जीवघेणा", "जानलेवा", "तातडीने", "तुरंत", "भीषण पूर",
        "अत्यंत धोकादायक", "गंभीर अपघात"
    ],
    4: [
        "deep pothole", "burst", "burst pipe", "spark", "accident", "hospital", "dead animal",
        "खूप मोठा खड्डा", "मोठा खड्डा", "बड़ा गड्ढा", "पाईप फुटला", "पाइप फट गया",
        "अपघात", "दुर्घटना", "दवाखाना", "रुग्णालय", "अस्पताल", "मरा हुआ जानवर"
    ],
    3: [
        "for three days", "for 3 days", "repeatedly", "overflowing", "no water",
        "तीन दिवसांपासून", "दोन दिवसांपासून", "३ दिवसांपासून", "तीन दिन से", "पानी नहीं",
        "पाणी नाही", "वारंवार", "तुंबले", "उतू चालले"
    ]
}

# Domain vocabulary enrichment to ensure scikit-learn TF-IDF aligns perfectly
CIVIC_KEYWORD_EXPANSIONS = {
    "pothole": [
        "खड्डा", "खड्डे", "गड्ढा", "गड्ढे", "खड्ड्यांमुळे", "खडडे", "गड्डे",
        "खड्ड्यात", "pothole", "gaddha", "khadda", "crater"
    ],
    "water leak": [
        "पाणी गळती", "पानी रिसाव", "पाईप", "पाइप", "गळती", "लीकेज", "पाईपलाईन", "पाइपलाइन", 
        "जलवाहिनी", "नळ", "नल गळती", "पेयजल", "water leak", "pipeline burst"
    ],
    "streetlight": [
        "स्ट्रीटलाईट", "स्ट्रीट लाईट", "पथदिवा", "पथदिवे", "अंधार", "अंधेरा", 
        "दिवा बंद", "लाईट पोल", "विजेचा खांब", "बिजली पोल", "बत्ती", "streetlight", "dark street"
    ],
    "garbage": [
        "कचरा", "कचराकुंडी", "घाण", "दुर्गंधी", "डस्टबिन", "कचरा गाडी", "कचरागाडी", 
        "सफाई", "उकिरडा", "कचरे", "बदबू", "दुर्गंध", "कचऱ्याचे साम्राज्य", "garbage", "waste dump", "kachra"
    ]
}


def detect_language(text: str) -> str:
    """
    Detect whether the input text is primarily English, Hindi, Marathi,
    or Multilingual/Romanized.
    """
    clean = text.strip()
    if not clean:
        return "Unknown"

    has_devanagari = bool(re.search(r'[\u0900-\u097F]', clean))
    if not has_devanagari:
        return "English / Romanized"

    # Distinguish Marathi vs Hindi markers
    marathi_markers = [
        "आहे", "नाही", "झाला", "झाली", "पडला", "पडली", "रस्त्यावर", "दिवे", "पाणी", 
        "खड्डा", "गळती", "कचऱ्याचे", "दिवसांपासून", "येथे", "होत", "आहेत"
    ]
    hindi_markers = [
        "है", "नहीं", "हुआ", "हुई", "गया", "गई", "सड़क", "बिजली", "पानी", 
        "गड्ढा", "कचरा", "दिनों", "यहाँ", "हो", "रहा", "रहे", "थी"
    ]

    m_score = sum(1 for m in marathi_markers if m in clean)
    h_score = sum(1 for h in hindi_markers if h in clean)

    if m_score > h_score:
        return "Marathi"
    elif h_score > m_score:
        return "Hindi"
    return "Hindi / Marathi (Devanagari)"


def check_multilingual_electrical_hazard(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if the text contains any electrical hazard keywords in English, Hindi, or Marathi.
    """
    lower = text.lower()
    for hazard in MULTILINGUAL_ELECTRICAL_HAZARDS:
        if hazard in lower or hazard in text:
            return True, f"Safety Escalation: Electrical hazard term '{hazard}' detected. Immediate electrical dispatch required."
    return False, None


CATEGORY_EXPANSION_TOKENS = {
    "pothole": "pothole road crater deep gaddha khadda",
    "water leak": "water leak pipeline burst drinking tap gushing",
    "streetlight": "streetlight dark street pole lamp light",
    "garbage": "garbage waste uncollected rotting kachra dump bins",
}


def normalize_multilingual_text(text: str) -> Dict[str, Any]:
    """
    Produce a normalized text representation for ML triage while preserving
    the original citizen submission.
    
    1. Transliterates Devanagari to Romanized text via AnyAscii.
    2. Injects English domain category keywords if Devanagari civic terms match.
    3. Checks for multilingual electrical hazards.
    """
    clean_text = text.strip()
    detected_lang = detect_language(clean_text)
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', clean_text))

    hazard_detected, hazard_warning = check_multilingual_electrical_hazard(clean_text)

    injected_tokens = []
    # Identify civic category domain clues
    for category, terms in CIVIC_KEYWORD_EXPANSIONS.items():
        for term in terms:
            if term in clean_text or term in clean_text.lower():
                injected_tokens.append(CATEGORY_EXPANSION_TOKENS.get(category, category))
                break

    # If text is in Devanagari, transliterate and enrich
    if has_devanagari:
        ascii_translit = anyascii.anyascii(clean_text)
        # Create an augmented text for the TF-IDF classifier
        enriched_parts = [clean_text, ascii_translit]
        if injected_tokens:
            enriched_parts.append(" ".join(injected_tokens))
        triage_text = " ".join(enriched_parts)
    else:
        triage_text = clean_text

    return {
        "original_text": clean_text,
        "triage_text": triage_text,
        "detected_language": detected_lang,
        "has_devanagari": has_devanagari,
        "is_electrical_hazard": hazard_detected,
        "hazard_warning": hazard_warning,
        "domain_hints": injected_tokens,
    }

