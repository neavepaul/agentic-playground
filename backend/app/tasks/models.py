from copy import deepcopy
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskContext(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    summary: str = "Starting task."
    cycle_count: int = 0
    tool_count: int = 0
    critic_count: int = 0
    current_plan: str = ""
    robot_status: dict = Field(default_factory=dict)
    discoveries: dict = Field(default_factory=dict)
    action_history: list[dict] = Field(default_factory=list)
    critic_feedback: list[dict] = Field(default_factory=list)
    explorer_reports: list[str] = Field(default_factory=list)

    def record(self, tool: str, arguments: dict, result: dict) -> None:
        entry = {"tool": tool, "arguments": arguments, **result}
        self.action_history.append(entry)
        if not result["success"]:
            return
        obs = result["observation"]
        if tool == "get_status":
            self.robot_status = deepcopy(obs)
        elif tool == "move_to":
            self.robot_status["room"] = obs["room"]
        elif tool == "look":
            self.discoveries["room:" + obs["room"]] = entry
        elif tool == "talk_to":
            self.discoveries["conversation:" + result["evidence_id"]] = entry
        elif tool in {"give", "drop", "pick_up"}:
            self.discoveries["object:" + arguments["object"]] = entry
            inventory = self.robot_status.setdefault("inventory", [])
            if tool == "pick_up" and arguments["object"] not in inventory:
                inventory.append(arguments["object"])
            elif tool != "pick_up" and arguments["object"] in inventory:
                inventory.remove(arguments["object"])

    def compact(self) -> dict:
        # Bounded task memory: latest room/object facts, recent conversations/actions.
        facts = [v for k, v in self.discoveries.items() if not k.startswith("conversation:")]
        conversations = [v for k, v in self.discoveries.items() if k.startswith("conversation:")]
        observed_rooms = {v["observation"]["room"] for v in facts if v["tool"] == "look"}
        known_exits = {room for v in facts if v["tool"] == "look"
                       for room in v["observation"]["connections"]}
        recent = self.action_history[-8:]
        recent_ids = {entry["evidence_id"] for entry in recent}
        older_discoveries = [entry for entry in facts + conversations[-8:]
                             if entry["evidence_id"] not in recent_ids]
        return {"goal": self.goal, "current_plan": self.current_plan,
                "robot_status": self.robot_status, "discoveries": older_discoveries,
                "known_but_unobserved_rooms": sorted(known_exits - observed_rooms),
                "recent_actions": recent,
                "critic_feedback": self.critic_feedback[-2:],
                "explorer_reports_unverified": self.explorer_reports[-2:],
                "cycle_count": self.cycle_count, "tool_count": self.tool_count}

    def public(self) -> dict:
        return self.model_dump(exclude={"action_history", "discoveries", "critic_feedback", "explorer_reports"})
