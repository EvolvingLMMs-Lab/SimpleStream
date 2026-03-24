"""OVO-Bench constants and scoring helpers for simpleStream code release."""

import re

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

# ---------------------------------------------------------------------------
# Prompt templates (from official OVO-Bench)
# ---------------------------------------------------------------------------
BR_PROMPT_TEMPLATE = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response.
"""

REC_PROMPT_TEMPLATE = """
You're watching a video in which people may perform a certain type of action repetively.
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one.
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

SSR_PROMPT_TEMPLATE = """
You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures:
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT_TEMPLATE = """
You're responsible of answering questions based on the video content.
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

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
