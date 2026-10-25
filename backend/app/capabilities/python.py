"""Small Python programs in Monty's restricted interpreter, never host exec()."""
import json
from pathlib import Path
import sys


async def run_python(code: str, data: dict, seconds: float = 3) -> dict:
    from pydantic_monty import AsyncMonty

    if len(json.dumps(data, ensure_ascii=False)) > 32000:
        raise ValueError("Python input data exceeds 32,000 characters.")
    output: list[str] = []
    size = 0

    def capture(_stream: str, chunk: str) -> None:
        nonlocal size
        size += len(chunk)
        if size > 8000:
            raise ValueError("Python output exceeds 8,000 characters. Print a smaller result.")
        output.append(chunk)

    binary = Path(sys.prefix) / ("Scripts/monty.exe" if sys.platform == "win32" else "bin/monty")
    if not binary.is_file():
        raise ValueError("Python runner is unavailable. Install backend/requirements.txt in the backend environment.")
    # No mounts, host functions, environment, network or hardware are exposed.
    # A fresh worker/session per call also prevents state leaking across tasks.
    async with AsyncMonty(binary_path=str(binary), min_processes=1, max_processes=1,
                          request_timeout=seconds + 3) as pool:
        async with pool.checkout(limits={"max_feed_duration_secs": seconds,
                                        "max_turn_duration_secs": seconds,
                                        "max_memory": 64 * 1024 * 1024,
                                        "max_recursion_depth": 100}) as session:
            value = await session.feed_run(code, inputs={"data": data}, print_callback=capture)
    result = repr(value)
    return {"stdout": "".join(output), "result": result[:8000], "result_truncated": len(result) > 8000,
            "runtime": "Monty restricted Python (no files, network, shell or robot access)"}
