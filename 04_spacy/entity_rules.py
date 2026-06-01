"""
04_spacy/entity_rules.py
-------------------------
Domain entity vocabulary for the facility support task.
Used by spaCy PhraseMatcher and the entity-augmented classifier.

Each entity group maps to a binary feature: "is this concept mentioned?"
Import ENTITY_GROUPS from here; do not run directly.
"""

from __future__ import annotations

# Each group becomes one binary feature: 1 if any phrase matches, 0 otherwise.
ENTITY_GROUPS: dict[str, list[str]] = {

    # Equipment / systems
    "eq_hvac": [
        "HVAC", "hvac", "air conditioning", "air conditioner", "AC unit",
        "heating system", "cooling system", "heat pump", "ventilation",
        "thermostat", "boiler", "furnace", "ductwork",
    ],
    "eq_plumbing": [
        "plumbing", "pipe", "pipes", "leak", "leaking", "faucet", "tap",
        "drain", "toilet", "sewage", "water heater", "burst pipe",
    ],
    "eq_electrical": [
        "electrical", "electricity", "wiring", "circuit", "breaker",
        "outlet", "socket", "sparks", "power outage", "lighting",
    ],
    "eq_elevator": [
        "elevator", "lift", "escalator",
    ],
    "eq_cleaning_general": [
        "cleaning", "cleaner", "clean", "mop", "vacuum", "sweep",
    ],
    "eq_cleaning_specialized": [
        "deep clean", "carpet", "upholstery", "window washing", "pressure wash",
        "mold remediation", "mould remediation", "sanitize", "disinfect",
        "steam clean", "floor polish", "strip and wax",
    ],

    # Urgency markers
    "urgency_high": [
        "immediately", "right away", "as soon as possible", "ASAP", "urgent",
        "urgently", "emergency", "critical", "cannot wait", "right now",
        "today", "tonight", "this morning", "this afternoon",
    ],
    "urgency_medium": [
        "soon", "at your earliest", "within the week", "within a few days",
        "timely", "upcoming event", "before the event", "follow up",
    ],
    "urgency_low": [
        "when convenient", "no rush", "whenever possible", "at your leisure",
        "curious", "wondering", "inquire", "information",
    ],

    # Safety / hazard
    "safety_hazard": [
        "safety", "hazard", "dangerous", "health risk", "unsafe",
        "mold", "mould", "toxic", "fumes", "contamination", "allergen",
        "injury", "accident", "slip", "flood", "flooding",
    ],

    # Sentiment markers
    "sentiment_positive": [
        "appreciate", "excellent", "outstanding", "wonderful", "fantastic",
        "very satisfied", "highly satisfied", "great service", "commend",
        "impressed", "delighted", "exceptional", "always been happy",
    ],
    "sentiment_negative": [
        "upset", "frustrated", "disappointed", "unacceptable", "poor service",
        "not satisfied", "not happy", "terrible", "horrible", "angry",
        "never came", "no one showed", "still not fixed", "ignored",
    ],

    # Service type intent
    "intent_scheduling": [
        "schedule", "reschedule", "booking", "appointment", "timing",
        "bi-weekly", "bi weekly", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "cleaning day", "service date",
    ],
    "intent_inquiry": [
        "inquire", "inquiry", "question", "more information", "more details",
        "tell me about", "what are your", "how do you", "pricing", "cost",
        "quote", "availability", "first time", "new client",
    ],
    "intent_complaint": [
        "complaint", "complain", "not satisfied", "poor quality", "redo",
        "refund", "compensation", "apology", "missed spot", "still dirty",
        "did not clean", "no one responded", "unreturned call",
    ],
    "intent_training": [
        "training", "guide", "guidance", "how to", "instruct",
        "show me how", "walk me through", "best practice", "procedure",
    ],

    # Sustainability
    "sustainability": [
        "eco-friendly", "green", "sustainable", "sustainability",
        "biodegradable", "non-toxic", "organic", "carbon footprint",
        "renewable", "recycling", "energy efficient",
    ],
}

# Flat list of all feature names (column order for the feature matrix)
FEATURE_NAMES = list(ENTITY_GROUPS.keys())
