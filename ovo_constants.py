"""OVO-Bench constants and scoring helpers for simpleStream code release."""

import re

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
BR_PROMPT_TEMPLATE = (
    "{0}\n"
    "Options: {1}\n"
    "Only give the best option's letter directly."
)

REC_PROMPT_TEMPLATE = (
    "{0}\n"
    "Only give a number as answer."
)

SSR_PROMPT_TEMPLATE = (
    "Is this person performing the tutorial step: {0}\n"
    "Answer Yes or No only."
)

CRR_PROMPT_TEMPLATE = (
    "{0}\n"
    "Is there enough information in the provided video to answer the question? "
    "Answer Yes or No only."
)

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def extract_br_answer(response):
    """Extract A/B/C/D answer from response text."""
    if not response or not str(response).strip():
        return None
    s = str(response).strip()
    m = re.search(r"\b([A-D])\b", s.upper())
    if m:
        return m.group(1)
    m = re.search(r"\b([1-4])\b", s)
    if m:
        return chr(64 + int(m.group(1)))
    return None


def score_br(response, gt):
    """Score a backward/realtime multiple-choice answer."""
    pred = extract_br_answer(response)
    return 1 if (pred is not None and pred.upper() == gt.upper()) else 0


def score_rec(response, gt_count):
    """Score a REC (repetition counting) answer."""
    if response is None or not str(response).strip():
        return 0
    nums = re.findall(r"\d+", str(response))
    return int("".join(nums) == str(gt_count)) if nums else 0


def score_yesno(response, gt_type):
    """Score a SSR/CRR yes/no answer. gt_type: 0=No, 1=Yes."""
    if response is None or not str(response).strip():
        return 0
    s = str(response).strip().upper()
    if (s == "N" or "NO" in s) and gt_type == 0:
        return 1
    if (s == "Y" or "YES" in s) and gt_type == 1:
        return 1
    return 0
