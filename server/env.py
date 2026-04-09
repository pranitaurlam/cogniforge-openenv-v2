from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .tasks import get_grader, clamp


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
    score: float = Field(..., gt=0.0, lt=1.0, description="Score strictly between 0 and 1")
    reason: str = Field(..., description="Reason for the score")


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
        if self.current_index >= len(self.tickets):
            return self._observe(), Reward(score=0.01, reason="Episode already finished"), True, {}

        ticket = self.tickets[self.current_index]
        grader = get_grader(ticket["id"])

        if action.is_done:
            raw = grader.grade(action.model_dump(), ticket["ground_truth"])
            score = clamp(raw)
            self.current_index += 1
            self._state["processed_count"] += 1
            reward = Reward(score=score, reason=f"Evaluated task {ticket['id']}")
        else:
            reward = Reward(score=0.01, reason="Intermediate step")

        done = self.current_index >= len(self.tickets)
        return self._observe(), reward, done, {}

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
