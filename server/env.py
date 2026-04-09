from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Observation(BaseModel):
    ticket_id: str = Field(..., description="Unique ID of the ticket")
    content: str = Field(..., description="Content of the support ticket")
    current_status: str = Field(..., description="Current ticket status")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    category: Optional[str] = Field(None, description="Ticket category: Account, Billing, Technical, Sales")
    priority: Optional[str] = Field(None, description="Priority: Low, Medium, High, Urgent")
    draft_response: Optional[str] = Field(None, description="Draft reply to the customer")
    is_done: bool = Field(False, description="Whether the agent is done with this ticket")


class Reward(BaseModel):
    score: float = Field(..., description="Reward score strictly between 0 and 1")
    reason: str = Field(..., description="Reason for the score")


def _safe_score(raw: float) -> float:
    """Hard clamp — score is always strictly between 0 and 1, no exceptions."""
    try:
        s = float(raw)
    except Exception:
        s = 0.5
    return max(0.01, min(0.99, s))


class SupportEnv:
    def __init__(self, tickets: List[Dict[str, Any]]):
        self.tickets = tickets
        self.current_index = 0
        self._state = {"processed_count": 0, "session_id": "sess_001"}

    def reset(self) -> Observation:
        self.current_index = 0
        self._state["processed_count"] = 0
        return self._observe()

    def step(self, action: Action):
        try:
            if self.current_index >= len(self.tickets):
                return self._observe(), Reward(score=0.5, reason="Episode already finished"), True, {}

            ticket = self.tickets[self.current_index]

            from .tasks import get_grader
            grader = get_grader(ticket["id"])
            raw = grader.grade(action.model_dump(), ticket["ground_truth"])
            score = _safe_score(raw)

            if action.is_done:
                self.current_index += 1
                self._state["processed_count"] += 1
                reason = f"Evaluated task {ticket['id']}"
            else:
                reason = "Intermediate step"

            done = self.current_index >= len(self.tickets)
            return self._observe(), Reward(score=score, reason=reason), done, {}

        except Exception:
            return self._observe(), Reward(score=0.5, reason="Error during evaluation"), False, {}

    def state(self) -> Dict[str, Any]:
        return self._state

    def _observe(self) -> Observation:
        if self.current_index < len(self.tickets):
            t = self.tickets[self.current_index]
            return Observation(
                ticket_id=t["id"],
                content=t["content"],
                current_status="PENDING",
                metadata=t.get("metadata", {}),
            )
        return Observation(ticket_id="END", content="END", current_status="DONE")
