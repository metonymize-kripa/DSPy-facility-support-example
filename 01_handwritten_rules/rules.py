"""
01_handwritten_rules/rules.py
------------------------------
All rule definitions for the handwritten classifier.
No I/O here — only pure functions.
Import this from classify.py.

Design follows the human expert decision cascade from CONCEPTUAL_FRAMEWORK.md:
  Step 1: Subject line scan
  Step 2: Explicit urgency markers
  Step 3: Service type vocabulary lookup
  Step 4: Message arc reading (sentiment)
  Step 5: Default rules
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Keyword vocabularies
# ---------------------------------------------------------------------------

# --- Urgency ---
URGENCY_HIGH_SUBJECT = re.compile(
    r"\b(urgent|emergency|immediate|asap|critical|broken|failed|failure|"
    r"not working|out of order|hazard|danger|safety)\b", re.I)

URGENCY_HIGH_BODY = re.compile(
    r"\b(immediately|right away|as soon as possible|asap|urgent|urgently|"
    r"emergency|cannot wait|can't wait|right now|today|tonight|this morning|"
    r"this evening|safety|hazard|dangerous|health risk|serious|severe|"
    r"unbearable|impossible to|cannot function|critical|life-threatening|"
    r"flooding|flooded|gas leak|electrical|sparks|fire)\b", re.I)

URGENCY_MEDIUM_BODY = re.compile(
    r"\b(soon|within (a |the )?(week|few days|next|coming)|"
    r"at your earliest|timely|upcoming event|guests arriving|"
    r"before (the|next|our)|schedule|reschedule|follow.?up|"
    r"not functioning (properly|well|correctly)|inconsistent|"
    r"continuing issue|recurring|ongoing|still (not|hasn't))\b", re.I)

URGENCY_LOW_BODY = re.compile(
    r"\b(inquir|question|information|when (convenient|possible|you (can|have time))|"
    r"no rush|not (urgent|pressing|immediate)|whenever|at leisure|"
    r"just (want|wanted|checking)|curious|wondering|looking into)\b", re.I)

# --- Sentiment ---
SENTIMENT_NEGATIVE = re.compile(
    r"\b(upset|frustrated|disappointed|unacceptable|terrible|horrible|poor|"
    r"awful|angry|furious|disgusted|outraged|appalled|shocked|cannot believe|"
    r"failed|failure|neglected|ignored|no.?one (picked up|responded|called back)|"
    r"still (not|hasn't)|despite|however (I must|I need to|I want to)|"
    r"not (satisfied|happy|pleased|acceptable)|very (upset|disappointed|frustrated)|"
    r"completely (unacceptable|unprofessional)|lack of|never (came|showed|fixed))\b", re.I)

SENTIMENT_POSITIVE = re.compile(
    r"\b(excellent|outstanding|exceptional|wonderful|fantastic|amazing|"
    r"very (happy|pleased|satisfied|impressed)|highly (satisfied|recommend)|"
    r"great (job|work|service)|truly (appreciate|grateful|thankful)|"
    r"commend|impressed|delighted|overjoyed)\b", re.I)

# Polite opener + complaint body: the register trap
POSITIVE_OPENER = re.compile(
    r"\b(appreciate|commend|always (been|found)|great service|wonderful|"
    r"exceptional|satisfied|pleased|happy)\b", re.I)

COMPLAINT_BODY = re.compile(
    r"\b(however|but (I|we)|unfortunately|issue|problem|concern|"
    r"not (working|functioning|satisfied|fixed|resolved)|still|"
    r"disappointed|need (to|your)|request|require)\b", re.I)

# --- Categories ---
CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    "emergency_repair_services": re.compile(
        r"\b(emergency|broken|not working|out of order|failed|failure|"
        r"repair (needed|required|asap|immediately)|flooding|flood|"
        r"gas leak|electrical (fault|problem|issue)|sparks|fire|"
        r"burst (pipe|water)|sewage|structural damage|"
        r"immediate (repair|attention|fix))\b", re.I),

    "routine_maintenance_requests": re.compile(
        r"\b(routine|regular|scheduled|maintenance|plumbing|pipe|leak|"
        r"faucet|tap|drain|toilet|hvac|heating|cooling|air (conditioning|con)|"
        r"thermostat|boiler|elevator|lift|light(ing)?|bulb|door|lock|"
        r"inspection|check(ing)? (the|on)|servic(e|ing)|tune.?up|"
        r"filter (replacement|change)|annual|monthly|periodic)\b", re.I),

    "cleaning_services_scheduling": re.compile(
        r"\b(schedul|reschedul|book(ing)?|appointment|slot|timing|"
        r"(cleaning|service) (time|date|day|schedule|calendar)|"
        r"bi.?weekly|weekly|monday|tuesday|wednesday|thursday|friday|"
        r"adjust (the |our )?(schedule|timing|cleaning)|"
        r"change (our |the )?(schedule|timing|cleaning day))\b", re.I),

    "specialized_cleaning_services": re.compile(
        r"\b(deep clean|carpet|upholstery|window wash|pressure wash|"
        r"mold|mould|remediation|sanitiz|disinfect|steam clean|"
        r"strip and wax|floor (polish|buff|coat)|grout|tile clean|"
        r"post.?construction|move.?(in|out) clean|hood clean|"
        r"specialized|specialist clean)\b", re.I),

    "customer_feedback_and_complaints": re.compile(
        r"\b(complaint|dissatisfied|not (satisfied|happy|pleased)|"
        r"poor (service|quality|job)|substandard|below (standard|expectation)|"
        r"still dirty|not clean|missed (spot|area)|did not (clean|fix|show)|"
        r"redo|refund|compensation|apology|apologize|no.?one (came|showed up)|"
        r"feedback|review|concern about (the |your )?(service|team|quality)|"
        r"left a message|called (but|and) no|unreturned)\b", re.I),

    "quality_and_safety_concerns": re.compile(
        r"\b(quality|standard|not up to|below (standard|par)|"
        r"safety|safe|unsafe|hazard|dangerous|health (risk|concern|hazard)|"
        r"chemical|fume|smell|odor|mold|mould|contamina|"
        r"injury|accident|slip|fall|exposed|toxic|allergen|"
        r"concern(ed)? about (safety|quality|health|standard))\b", re.I),

    "sustainability_and_environmental_practices": re.compile(
        r"\b(eco.?friendly|green|sustainable|sustainability|environment(al)?|"
        r"biodegradable|non.?toxic|organic|chemical.?free|"
        r"carbon (footprint|neutral|offset)|renewable|recycle|recycling|"
        r"energy (efficient|saving|consumption)|water (conservation|saving)|"
        r"certif(ied|ication) (green|eco|environment))\b", re.I),

    "training_and_support_requests": re.compile(
        r"\b(train(ing)?|support|guid(e|ance|ance)|tutorial|how (to|do I)|"
        r"instruct|help (me|us) (with|understand|learn)|"
        r"show (me|us) how|walk.?(me|us) through|best practice|"
        r"procedure|protocol|onboard)\b", re.I),

    "facility_management_issues": re.compile(
        r"\b(facility management|building management|property management|"
        r"manag(er|ement) (issue|problem|concern)|lease|rent|contract|"
        r"vendor|contractor|budget|invoice|billing|access (control|card|key)|"
        r"common area|shared space|tenant|landlord|building (issue|problem))\b", re.I),

    "general_inquiries": re.compile(
        r"\b(inquir(e|y|ing)|question(s)?|information|more (detail|info)|"
        r"tell me (more|about)|what (are|is) your|how (do|does)|"
        r"what (services|options|plans)|pricing|cost|quote|package|"
        r"available|availability|would like to know|looking (into|for information)|"
        r"first time|new (client|customer)|considering|thinking about)\b", re.I),
}

# Categories where subject-line matches are high-confidence
SUBJECT_PRIORITY_CATEGORIES = {
    "emergency_repair_services",
    "cleaning_services_scheduling",
    "specialized_cleaning_services",
}


# ---------------------------------------------------------------------------
# Classifier functions
# ---------------------------------------------------------------------------

def classify_urgency(message: str) -> str:
    """
    Cascade: first match wins.
    Checks subject line first, then body.
    """
    lines = message.strip().split("\n")
    subject = ""
    body = message

    # Extract subject line if present
    for line in lines[:3]:
        if line.lower().startswith("subject:"):
            subject = line[8:].strip()
            body = "\n".join(lines[1:])
            break

    # Step 1: Subject line urgent keywords
    if URGENCY_HIGH_SUBJECT.search(subject):
        return "high"

    # Step 2: Body explicit urgency markers
    if URGENCY_HIGH_BODY.search(body):
        return "high"

    # Step 3: Low-urgency inquiry signals (check before medium — avoid false mediums)
    if URGENCY_LOW_BODY.search(body) and not URGENCY_MEDIUM_BODY.search(body):
        return "low"

    # Step 4: Medium signals
    if URGENCY_MEDIUM_BODY.search(body):
        return "medium"

    # Step 5: Default — medium (most facility issues need timely attention)
    return "medium"


def classify_sentiment(message: str) -> str:
    """
    Register-aware sentiment: checks for polite-opener + complaint-body pattern.
    """
    # Check for explicit negative language anywhere
    if SENTIMENT_NEGATIVE.search(message):
        return "negative"

    # Register trap: positive opener but complaint body
    first_300 = message[:300]
    rest = message[300:]
    if POSITIVE_OPENER.search(first_300) and COMPLAINT_BODY.search(rest):
        return "neutral"

    # Explicit positive language (not in opener context)
    if SENTIMENT_POSITIVE.search(message):
        return "positive"

    # Default — professional correspondence is neutral
    return "neutral"


def classify_categories(message: str) -> list[str]:
    """
    Non-exclusive — all matching patterns fire.
    Subject-line matches are high-confidence for priority categories.
    """
    lines = message.strip().split("\n")
    subject = ""
    body = message

    for line in lines[:3]:
        if line.lower().startswith("subject:"):
            subject = line[8:].strip()
            body = "\n".join(lines[1:])
            break

    active = []

    for cat, pattern in CATEGORY_PATTERNS.items():
        if cat in SUBJECT_PRIORITY_CATEGORIES and pattern.search(subject):
            active.append(cat)
        elif pattern.search(body):
            active.append(cat)

    # Fallback: if nothing matched, assign general_inquiries
    if not active:
        active = ["general_inquiries"]

    return active


def predict(message: str) -> dict:
    """Full prediction for one message. Returns dict compatible with shared.metrics."""
    return {
        "urgency": classify_urgency(message),
        "sentiment": classify_sentiment(message),
        "categories": classify_categories(message),
    }
