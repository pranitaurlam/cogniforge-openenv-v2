from typing import Dict, Any, List


def clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return max(0.01, min(0.99, float(score)))


class TaskGrader:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        raise NotImplementedError


class EasyGrader(TaskGrader):
    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        action_cat = (action.get("category") or "").strip().lower()
        truth_cat = (ground_truth.get("category") or "").strip().lower()
        return clamp(0.9 if action_cat == truth_cat else 0.1)


class MediumGrader(TaskGrader):
    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        score = 0.0
        if (action.get("category") or "").strip().lower() == (ground_truth.get("category") or "").strip().lower():
            score += 0.5
        if (action.get("priority") or "").strip().lower() == (ground_truth.get("priority") or "").strip().lower():
            score += 0.5
        return clamp(score if score > 0 else 0.1)


class HardGrader(TaskGrader):
    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        score = 0.0
        if (action.get("category") or "").strip().lower() == (ground_truth.get("category") or "").strip().lower():
            score += 0.2
        if (action.get("priority") or "").strip().lower() == (ground_truth.get("priority") or "").strip().lower():
            score += 0.2
        draft = (action.get("draft_response") or "").strip()
        if len(draft) > 15:
            score += 0.2
            keywords: List[str] = ground_truth.get("keywords") or []
            if keywords:
                hits = sum(1 for kw in keywords if kw.lower() in draft.lower())
                score += 0.4 * (hits / len(keywords))
            else:
                score += 0.4
        return clamp(score if score > 0 else 0.1)


class DefaultGrader(TaskGrader):
    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        return clamp(0.5)


TASKS_DATA = [
    {
        "id": "task_1_easy",
        "content": (
            "Subject: Login Issue\n\n"
            "Hi support, I can't log in to my account. I keep getting an "
            "'Invalid credentials' error even though I reset my password yesterday. "
            "Can you help?"
        ),
        "ground_truth": {
            "category": "Account",
            "priority": "Medium",
            "keywords": ["login", "credentials", "account", "password"],
        },
    },
    {
        "id": "task_2_medium",
        "content": (
            "Subject: Slow Performance & Critical Deadline\n\n"
            "Hello, the dashboard is extremely slow today. It takes 30 seconds to "
            "load any report. I have a client presentation in 2 hours and I need "
            "this working ASAP. Please look into this immediately."
        ),
        "ground_truth": {
            "category": "Technical",
            "priority": "High",
            "keywords": ["slow", "performance", "load", "critical", "deadline"],
        },
    },
    {
        "id": "task_3_hard",
        "content": (
            "Subject: Duplicate Subscription Charge\n\n"
            "I just checked my bank statement and I've been charged $49.99 twice "
            "for my 'Pro Plan' this month. I only have one active subscription. "
            "This is unacceptable, please refund the duplicated amount and ensure "
            "it doesn't happen again."
        ),
        "ground_truth": {
            "category": "Billing",
            "priority": "Urgent",
            "keywords": ["duplicate", "charge", "refund", "subscription", "Pro Plan"],
        },
    },
]


def get_grader(task_id: str) -> TaskGrader:
    tid = (task_id or "").lower()
    if "easy" in tid:
        return EasyGrader(task_id)
    if "medium" in tid:
        return MediumGrader(task_id)
    if "hard" in tid:
        return HardGrader(task_id)
    return DefaultGrader(task_id)
