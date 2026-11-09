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
"""Build real-text prompts of an exact token length from ShareGPT.

Text is the ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered) conversation
turns, concatenated in file order into one token stream. Each prompt is a
contiguous, non-overlapping slice of that stream, so every prompt is unique
and no two share a prefix. Each prompt is re-encoded the way the vLLM server
encodes a /v1/completions prompt (add_special_tokens=True, so <bos> counts)
and kept only if that gives exactly the target length.

Output: one JSONL per set, rows {"prompt": str, "prompt_tokens": int}.
Usage: build_dataset.py --model M --set name:isl:count [--set ...] --out-dir D
"""
import argparse
import json
import os
import sys

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--repo",
                    default="anon8231489123/ShareGPT_Vicuna_unfiltered")
    ap.add_argument("--file",
                    default="ShareGPT_V3_unfiltered_cleaned_split.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    path = hf_hub_download(a.repo, a.file, repo_type="dataset")
    convs = json.load(open(path))
    print(f"loaded {len(convs)} conversations from {a.repo}/{a.file}",
          flush=True)

    sets = []
    for s in a.set:
        name, isl, n = s.split(":")
        sets.append((name, int(isl), int(n)))
    bos = 1 if tok(
        "x", add_special_tokens=True).input_ids[0] == tok.bos_token_id else 0

    def stream():
        for c in convs:
            text = "\n\n".join(
                t.get("value", "") for t in c.get("conversations", [])
                if t.get("value"))
            if len(text) < 200:
                continue
            ids = tok(text + "\n\n", add_special_tokens=False).input_ids
            yield from ids

    it = stream()
    buf = []
    os.makedirs(a.out_dir, exist_ok=True)
    report = {
        "source": f"{a.repo}/{a.file}",
        "tokenizer": a.model,
        "bos_counted": bool(bos),
        "sets": {}
    }
    for name, isl, n in sets:
        rows, rejected = [], 0
        while len(rows) < n:
            need = isl - bos + 8
            while len(buf) < need:
                try:
                    buf.append(next(it))
                except StopIteration:
                    sys.exit(f"corpus exhausted at {name}: {len(rows)}/{n}")
            target = isl - bos
            ok = None
            for k in (target, target - 1, target + 1, target - 2, target + 2,
                      target - 3, target + 3):
                text = tok.decode(buf[:k], skip_special_tokens=True)
                if len(tok(text, add_special_tokens=True).input_ids) == isl:
                    ok = (text, k)
                    break
            if ok is None:
                rejected += 1
                del buf[:target]
                continue
            rows.append({"prompt": ok[0], "prompt_tokens": isl})
            del buf[:ok[1]]
        fn = os.path.join(a.out_dir, f"{name}.jsonl")
        with open(fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        uniq = len({r["prompt"] for r in rows})
        report["sets"][name] = {
            "isl": isl,
            "count": len(rows),
            "unique": uniq,
            "rejected_slices": rejected,
            "file": f"{name}.jsonl"
        }
        print(
            f"{name}: {len(rows)} prompts of {isl} tokens, {uniq} unique, "
            f"{rejected} slices rejected",
            flush=True)
    json.dump(report,
              open(os.path.join(a.out_dir, "dataset_report.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
