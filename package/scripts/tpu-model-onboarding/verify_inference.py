# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Send one inference request to a served model and report what came back.

Runs on the controller against a port `compute port-forward` has brought here,
the same way health_probe.py does.

A 200 is not enough. The completion must also be non-empty and must not be
degenerate: a single repeated character ("!!!!", what NaN logits decode to),
a phrase repeating back to back ("the the the"), text that compresses far
better than language does, or text made mostly of <unk>/<pad>/U+FFFD. Each
of those is a model that is not working, served by an engine that is.

The request is greedy (temperature 0) so the verdict does not change between
runs, and goes through the model's chat template where there is one (vLLM's
chat endpoint; the FastAPI recipe applies it server side), because an instruct
model handed a raw string can loop on its own and fail the repetition checks.

After a passing request, a known-answer question is asked. A wrong answer is
a warning, not a failure: small or base models can miss it. It is the only
part of this check that can notice fluent output that is wrong.

This is still a functional check, not a correctness one. Numerical parity is
the numerical-parity agent's job.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
import zlib

DEFAULT_PROMPT = "Hello! Briefly, what is a tensor?"
KNOWN_ANSWER_PROMPT = "What is the capital of France? Answer in one word."
KNOWN_ANSWER = "paris"

# Markers a broken tokenizer or NaN logits decode to. U+FFFD is what invalid
# UTF-8 byte sequences become.
GARBAGE_MARKERS = ("<unk>", "<pad>", "�")
MAX_GARBAGE_FRACTION = 0.2
# English compresses to roughly 1-1.5x with zlib at this length; loops go far
# higher. 2.4 is the threshold Whisper uses for the same purpose.
MAX_COMPRESSION_RATIO = 2.4
# Below this, zlib's fixed overhead makes the ratio meaningless.
MIN_COMPRESSION_BYTES = 40
MAX_NGRAM = 6
MAX_CONSECUTIVE_REPEATS = 4
OPENAI_ENGINES = ("vllm", "sglang")


def build_request(engine: str, model_id: str, prompt: str, max_tokens: int,
                  chat: bool) -> tuple[str, dict]:
    """Return (path, payload) for the engine's own API shape."""
    if engine in OPENAI_ENGINES:
        # vLLM and SGLang serve the OpenAI spec and require the model name in the body.
        if chat:
            return "/v1/chat/completions", {
                "model": model_id,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        return "/v1/completions", {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    return "/generate", {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0
    }


def extract_completion(engine: str, body: dict) -> str:
    """Pull the generated text out of whichever envelope the engine used."""
    if engine in OPENAI_ENGINES:
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message.get("content") or ""
        return choices[0].get("text") or ""
    # "result" is what the fastapi_server recipes return.
    for key in ("result", "generated_text", "text", "output", "completion",
                "response"):
        if isinstance(body.get(key), str):
            return body[key]
    return ""


def strip_prompt_echo(completion: str, prompt: str) -> str:
    """Drop the prompt if the server decoded it back along with the answer.

    Left in, the prompt's own words dilute every repetition check.
    """
    if not completion:
        return ""
    head = completion.lstrip()
    if prompt.strip() and head.startswith(prompt.strip()):
        return head[len(prompt.strip()):]
    return completion


def _longest_consecutive_repeat(tokens: list[str]) -> tuple[int, str]:
    """Most back-to-back copies of any n-gram (n <= MAX_NGRAM) in tokens."""
    best, best_gram = 1, ""
    for n in range(1, MAX_NGRAM + 1):
        for start in range(len(tokens) - n + 1):
            gram = tokens[start:start + n]
            count = 1
            pos = start + n
            while tokens[pos:pos + n] == gram:
                count += 1
                pos += n
            if count > best:
                best, best_gram = count, " ".join(gram)
    return best, best_gram


def degeneracy_failures(text: str) -> list[str]:
    """Reasons text is not something a working model produced; [] if none."""
    stripped = text.strip()
    if not stripped:
        return ["empty completion"]

    failures = []
    garbage = sum(stripped.count(m) * len(m) for m in GARBAGE_MARKERS)
    if garbage / len(stripped) >= MAX_GARBAGE_FRACTION:
        failures.append(f"{garbage}/{len(stripped)} characters are "
                        "<unk>/<pad>/U+FFFD")

    visible = re.sub(r"\s", "", stripped)
    if len(visible) >= 4 and len(set(visible)) == 1:
        failures.append(f"a single character repeated: {visible[0]!r}")

    encoded = stripped.encode("utf-8")
    if len(encoded) >= MIN_COMPRESSION_BYTES:
        ratio = len(encoded) / len(zlib.compress(encoded))
        if ratio > MAX_COMPRESSION_RATIO:
            failures.append(f"compression ratio {ratio:.2f} > "
                            f"{MAX_COMPRESSION_RATIO} (repetitive text)")

    tokens = re.findall(r"\w+|[^\w\s]+", stripped.lower())
    repeats, gram = _longest_consecutive_repeat(tokens)
    if repeats >= MAX_CONSECUTIVE_REPEATS:
        failures.append(f"{gram!r} repeated {repeats} times in a row")
    return failures


def post_json(url: str, payload: dict,
              timeout: int) -> tuple[int | None, str, str]:
    """POST payload; return (status, raw body, error)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            # Read it all: a body cut short is unparseable JSON, and that
            # would fail a healthy deployment. The report truncates on write.
            return resp.status, resp.read().decode("utf-8",
                                                   errors="replace"), ""
    except urllib.error.HTTPError as exc:
        return exc.code, "", (exc.read(2048).decode("utf-8", errors="replace")
                              if hasattr(exc, "read") else str(exc))
    except Exception as exc:  # noqa: BLE001 - reported in the payload
        return None, "", f"{type(exc).__name__}: {exc}"


def ask(port: int, engine: str, model_id: str, prompt: str, max_tokens: int,
        timeout: int) -> dict:
    """Send prompt and return what came back, prompt echo removed.

    vLLM and SGLang go through the chat endpoint first, and fall back to plain
    completions only when the model has no chat template to apply.
    """
    chat = engine in OPENAI_ENGINES
    while True:
        path, payload = build_request(engine, model_id, prompt, max_tokens,
                                      chat)
        url = f"http://localhost:{port}{path}"
        status, raw, error = post_json(url, payload, timeout)
        if chat and status == 400 and "chat template" in error.lower():
            chat = False
            continue
        break

    completion = ""
    if status == 200:
        try:
            completion = extract_completion(engine, json.loads(raw))
        except (ValueError, AttributeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "url": url,
        "status": status,
        "raw": raw,
        "error": error,
        "completion": strip_prompt_echo(completion, prompt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",
                        type=int,
                        required=True,
                        help="Local port (the forwarded one)")
    parser.add_argument("--engine",
                        default="custom_fastapi",
                        choices=["custom_fastapi", "vllm", "sglang"])
    parser.add_argument("--model-id",
                        default="",
                        help="Required by the vLLM/SGLang OpenAI endpoint")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-known-answer",
                        action="store_true",
                        help="Do not ask the known-answer question")
    parser.add_argument("--output", default="verification.json")
    args = parser.parse_args()

    if args.engine in OPENAI_ENGINES and not args.model_id:
        print(f"--model-id is required for the {args.engine} engine.",
              file=sys.stderr)
        return 2

    result = ask(args.port, args.engine, args.model_id, args.prompt,
                 args.max_tokens, args.timeout)
    status, completion = result["status"], result["completion"]
    if status != 200:
        failures = [f"HTTP status {status}"]
    elif result["error"]:
        failures = [f"unreadable response: {result['error']}"]
    else:
        failures = degeneracy_failures(completion)
    verified = not failures

    warnings = []
    known_answer = None
    if verified and not args.skip_known_answer:
        probe = ask(args.port, args.engine, args.model_id, KNOWN_ANSWER_PROMPT,
                    16, args.timeout)
        # A wrong answer is the model's; a failed request is the server's,
        # and that is not something a warning covers.
        if probe["status"] != 200 or probe["error"]:
            failures.append(
                f"known-answer request failed with HTTP {probe['status']}: "
                f"{probe['error'][:200]}")
            verified = False
        passed = (probe["status"] == 200
                  and KNOWN_ANSWER in probe["completion"].lower())
        known_answer = {
            "prompt": KNOWN_ANSWER_PROMPT,
            "expected": KNOWN_ANSWER,
            "completion": probe["completion"][:500],
            "status_code": probe["status"],
            "passed": passed,
        }
        if verified and not passed:
            warnings.append(
                f"known-answer question did not return {KNOWN_ANSWER!r}: "
                f"{probe['completion'][:200]!r}")

    report = {
        "verified": verified,
        "url": result["url"],
        "status_code": status,
        "engine": args.engine,
        "prompt": args.prompt,
        "completion": completion[:2000],
        "failures": failures,
        "warnings": warnings,
        "known_answer": known_answer,
        "raw_response": result["raw"][:2000],
        "error": result["error"][:1000],
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    if status == 200 and failures:
        print("Endpoint returned 200 but the completion failed: " +
              "; ".join(failures) + " -- the deployment is not verified.",
              file=sys.stderr)
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
