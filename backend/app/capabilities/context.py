"""Bounded prompt views of full tool evidence; audit records remain unchanged."""
from copy import deepcopy
import json

from .tools import DIGITAL_TOOLS


def evidence_view(context) -> tuple[list[dict], dict[str, str]]:
    aliases = {entry["evidence_id"]: f"e{index + 1}" for index, entry in enumerate(
        a for a in context.action_history if a["success"] and a["tool"] in DIGITAL_TOOLS - {"finish_answer"})}
    observations, remaining = [], 9000
    for entry in reversed(context.action_history):
        if entry["tool"] not in DIGITAL_TOOLS:
            continue
        item = {key: deepcopy(entry[key]) for key in ("tool", "success", "evidence_id", "observation", "error") if key in entry}
        item["evidence_id"] = aliases.get(entry["evidence_id"], entry["evidence_id"])
        obs = item.get("observation", {})
        if entry["tool"] == "read_webpage" and obs:
            text = obs.get("text", "")
            if len(text) > 2500:
                obs["text"] = text[:2500]
                obs["prompt_text_truncated"] = True
                obs["next_start_char"] = obs.get("start_char", 0) + 2500
            links = obs.get("links", [])
            obs["links"] = [{"title": link.get("title", "")[:100], "url": link["url"]}
                            for link in links if len(link.get("url", "")) <= 600][:3]
        elif entry["tool"] == "web_search" and obs:
            for result in obs.get("results", []):
                result["snippet"] = result.get("snippet", "")[:250]
        elif entry["tool"] == "run_python" and obs:
            for key in ("stdout", "result"):
                if len(obs.get(key, "")) > 2000:
                    obs[key] = obs[key][:2000]
                    obs["prompt_output_truncated"] = True
        if not entry["success"]:
            args = entry["arguments"]
            if entry["tool"] == "finish_answer":
                args = {"evidence_ids": [aliases.get(key, key) for key in args.get("evidence_ids", [])]}
            elif len(json.dumps(args, ensure_ascii=False)) > 4000:
                args = {"code_excerpt": args.get("code", "")[:2000], "note": "Large failed input omitted; simplify it."}
            item["arguments"] = args
        size = len(json.dumps(item, ensure_ascii=False))
        if size <= remaining:
            observations.append(item)
            remaining -= size
    return list(reversed(observations)), {alias: original for original, alias in aliases.items()}
