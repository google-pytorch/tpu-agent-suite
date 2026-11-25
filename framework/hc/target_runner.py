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
"""Target-side runner for Hillclimber serving, microbenchmark, and accuracy jobs.

Staged into a job's working directory beside canonical launcher scripts
(`serve_vllm.sh`, `serve_sglang.sh`, `eval_recipe.sh`, `forward_run.py`) to run
`serve-vllm`, `serve-sglang`, `eval-vllm`, or `run-microbenchmark` from sealed
`__hc_context.json`, candidate `__flags.json`, and optional `candidate.patch`.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
from typing import Any, Mapping, Sequence

Path = pathlib.Path

CONTEXT_FILENAME = "__hc_context.json"
FLAGS_FILENAME = "__flags.json"
PATCH_FILENAME = "candidate.patch"
VLLM_PATCHES_FILE = "vllm_patches.txt"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Canonical launcher filenames staged into the job working directory.
LAUNCHERS: dict[str, str] = {
    "vllm": "serve_vllm.sh",
    "sglang": "serve_sglang.sh",
    "eval": "eval_recipe.sh",
}


def _finite_float(val: Any) -> float | None:
    if (isinstance(val, (int, float)) and not isinstance(val, bool)
            and math.isfinite(val)):
        return float(val)
    return None


def _extract_metric(path: Path, metric: str, task: str = "") -> float | None:
    """Extract the finite scalar `metric` from an lm-eval `results_*.json` file."""
    data = _load_json_dict(path, required=True)
    raw_groups = data.get("groups")
    raw_results = data.get("results")
    groups = raw_groups if isinstance(raw_groups, dict) else {}
    results = raw_results if isinstance(raw_results, dict) else {}
    by_task: dict[str, Any] = {**results, **groups}

    if "/" in metric:
        task_name, key = metric.split("/", 1)
        entry = by_task.get(task_name)
        return (_finite_float(entry.get(key))
                if isinstance(entry, dict) else None)

    candidates: list[Any] = []
    if task and isinstance(by_task.get(task), dict):
        candidates.append(by_task[task])
    candidates.extend(v for v in groups.values() if isinstance(v, dict))
    candidates.extend(v for v in results.values() if isinstance(v, dict))
    for entry in candidates:
        if metric in entry:
            return _finite_float(entry[metric])
    return None


def _load_json_dict(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required config file missing: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _has_nonempty_patch(patch_file: Path) -> bool:
    return patch_file.is_file() and bool(patch_file.read_bytes().strip())


def _tokens(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


class _FlagParser(argparse.ArgumentParser):

    def error(self, message: str) -> Any:
        raise ValueError(message)


def _parse_flags(
    tokens: Sequence[str],
    *names: str,
) -> tuple[argparse.Namespace, list[str]]:
    parser = _FlagParser(add_help=False, allow_abbrev=False)
    for name in names:
        parser.add_argument(name, default=None)
    return parser.parse_known_args([str(t) for t in tokens])


def _resolve_model(workdir: Path, ctx: Mapping[str, Any]) -> str:
    weights_file = workdir / "weights_path.txt"
    if weights_file.is_file():
        val = weights_file.read_text(encoding="utf-8").strip()
        if val:
            return val
    for key in ("weights_path", "model", "target_model"):
        raw = ctx.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _job_env(flags: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env_map = flags.get("env") if isinstance(flags.get("env"), dict) else {}
    for key, val in sorted(env_map.items()):
        if not _ENV_KEY_RE.fullmatch(str(key)):
            raise ValueError(f"invalid environment variable name: {key!r}")
        env[str(key)] = ("true" if val is True else
                         "false" if val is False else str(val))
    return env


def _launcher(workdir: Path, kind: str) -> Path:
    staged = workdir / LAUNCHERS[kind]
    if staged.is_file():
        return staged
    raise FileNotFoundError(f"{LAUNCHERS[kind]} was not staged into {workdir}")


def serving_args(workdir: Path, *, engine: str,
                 cli_args: Sequence[str]) -> list[str]:
    """Build the argument list for `serve_vllm.sh` or `serve_sglang.sh`."""
    ctx = _load_json_dict(workdir / CONTEXT_FILENAME, required=True)
    flags = _load_json_dict(workdir / FLAGS_FILENAME)
    cand_server_args = _tokens(flags.get("server_args"))
    cand_known, _ = _parse_flags(
        cand_server_args,
        "--model",
        "--model-path",
        "--served-model-name",
    )
    if ((cand_server_args and not cand_server_args[0].startswith("-"))
            or "--" in cand_server_args
            or any(v is not None for v in vars(cand_known).values())):
        raise ValueError(
            f"{FLAGS_FILENAME}.server_args cannot override model or --served-model-name"
        )
    combined = _tokens(ctx.get("base_server_args")) + cand_server_args
    cli = [str(a) for a in cli_args]
    if _parse_flags(cli, "--port")[0].port is not None:
        _, combined = _parse_flags(combined, "--port")
    combined += cli

    model = _resolve_model(workdir, ctx)
    served_name = str(ctx.get("target_model") or model).strip()
    positional_model = bool(combined and not combined[0].startswith("-"))
    model_flags = ("--model", ) if engine == "vllm" else ("--model-path",
                                                          "--model")
    known_model, _ = _parse_flags(combined, *model_flags)

    args: list[str] = []
    if not positional_model and all(v is None
                                    for v in vars(known_model).values()):
        if not model:
            raise ValueError(
                f"no target_model or model specified in {workdir / CONTEXT_FILENAME}"
            )
        args += [model] if engine == "vllm" else ["--model-path", model]
    args += combined
    known_final, _ = _parse_flags(args, "--served-model-name", "--port")
    if served_name and known_final.served_model_name is None:
        args += ["--served-model-name", served_name]
    if known_final.port is None:
        args += ["--port", "8000"]
    if engine == "vllm":
        patch, patches_list = workdir / PATCH_FILENAME, workdir / VLLM_PATCHES_FILE
        if _has_nonempty_patch(patch) and not patches_list.is_file():
            patches_list.write_text(f"{PATCH_FILENAME}\n", encoding="utf-8")
    return args


def _first_patched_relpath(patch_file: Path) -> str | None:
    for line in patch_file.read_text(encoding="utf-8",
                                     errors="replace").splitlines():
        if line.startswith(("+++ ", "--- ")):
            raw = line[4:].split("\t", 1)[0].strip()
            if raw == "/dev/null":
                continue
            parts = raw.split("/", 1)
            rel = parts[1] if len(parts) == 2 and parts[0] in ("a",
                                                               "b") else raw
            if rel:
                return rel
    return None


_SHELL_NAMES = frozenset({"bash", "sh", "dash", "zsh", "ksh"})


def _resolve_python() -> str:
    raw = os.path.expanduser(
        os.environ.get("TPU_AGENT_SUITE_PYTHON", "").strip())
    if not raw or Path(raw).name in _SHELL_NAMES:
        return "python3"
    return raw


def _package_patch_root(package_name: str, patch_file: Path) -> Path | None:
    pkg_dir: Path | None = None
    py_override = _resolve_python()
    py_bin = shutil.which(py_override) if py_override != "python3" else None
    if py_bin:
        out = subprocess.run(
            [
                py_bin,
                "-c",
                (f"import importlib.util as u; "
                 f"s = u.find_spec({package_name!r}); "
                 f"print(list(s.submodule_search_locations)[0] "
                 f"if s and s.submodule_search_locations else '')"),
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if out:
            pkg_dir = Path(out).resolve()
    if pkg_dir is None:
        spec = importlib.util.find_spec(package_name)
        if spec is not None and spec.submodule_search_locations:
            pkg_dir = Path(list(spec.submodule_search_locations)[0]).resolve()
    if pkg_dir is None or not pkg_dir.is_dir():
        return None
    rel = _first_patched_relpath(patch_file)
    if rel:
        for cand in (pkg_dir.parent, pkg_dir, pkg_dir.parent.parent):
            if (cand / rel).exists():
                return cand
    return pkg_dir.parent


def apply_patch(workdir: Path, target_dir: Path | None = None) -> None:
    """Apply `candidate.patch` to installed `sglang` once; fail loudly on error."""
    patch_file = workdir / PATCH_FILENAME
    if not _has_nonempty_patch(patch_file):
        return
    dest = target_dir
    if dest is None:
        env_root = os.environ.get("HC_PATCH_TARGET_DIR", "").strip()
        dest = Path(env_root) if env_root else _package_patch_root(
            "sglang", patch_file)
    if dest is None or not dest.is_dir():
        raise RuntimeError(
            "candidate.patch is staged but no patch target was found: set "
            "HC_PATCH_TARGET_DIR, or install sglang where this Python sees it")
    patch_abs = str(patch_file.resolve())
    reversed_ok = subprocess.run(
        ["git", "-C",
         str(dest), "apply", "-p1", "-R", "--check", patch_abs],
        capture_output=True,
        check=False,
    ).returncode == 0
    if reversed_ok:
        return
    subprocess.run(
        ["git", "-C", str(dest), "apply", "-p1", patch_abs], check=True)


def serve(workdir: Path, engine: str, cli_args: Sequence[str]) -> int:
    """Replace this process with the shipped launcher for `engine`."""
    if engine == "sglang":
        apply_patch(workdir)
    flags = _load_json_dict(workdir / FLAGS_FILENAME)
    env = _job_env(flags)
    launcher = _launcher(workdir, engine)
    argv = [
        "bash",
        str(launcher), *serving_args(workdir, engine=engine, cli_args=cli_args)
    ]
    os.chdir(workdir)
    os.execvpe("bash", argv, env)
    return 1  # unreachable


def _default_eval_task(ctx: Mapping[str, Any]) -> str:
    want = str(ctx.get("accuracy_metric") or "").strip()
    inferred = want.split("/", 1)[0].strip() if "/" in want else ""
    return str(ctx.get("eval_tasks") or inferred or "mmlu_pro").strip()


def eval_args(workdir: Path, cli_args: Sequence[str]) -> list[str]:
    """Build `eval_recipe.sh` `KEY=VALUE` arguments and stage command files."""
    ctx = _load_json_dict(workdir / CONTEXT_FILENAME, required=True)
    flags = _load_json_dict(workdir / FLAGS_FILENAME)
    model = _resolve_model(workdir, ctx)
    served_name = str(ctx.get("target_model") or model or "model").strip()
    task = _default_eval_task(ctx)
    limit = str(ctx.get("eval_limit") or "100").strip()
    server_tokens = _tokens(ctx.get("base_server_args")) + _tokens(
        flags.get("server_args"))
    port = (str(ctx.get("server_port") or "").strip()
            or _parse_flags(server_tokens, "--port")[0].port or "8000")

    args = [
        f"SERVED_MODEL_NAME={served_name}",
        f"EVAL_TASKS={task}",
        f"EVAL_LIMIT={limit}",
        f"SERVER_PORT={port}",
    ]
    for tok in cli_args:
        if "=" in tok:
            key, val = tok.split("=", 1)
            if key:
                args = [a for a in args if not a.startswith(f"{key}=")]
                if key == "SERVER_PORT" and val:
                    port = val
        args.append(tok)

    python = _resolve_python()
    entry_ref = (Path(__file__).name if
                 (workdir / Path(__file__).name).is_file() else str(
                     Path(__file__).resolve()))
    server_cmd = workdir / "vllm_server_cmd.txt"
    if not server_cmd.is_file():
        server_cmd.write_text(
            f"exec {shlex.quote(python)} {shlex.quote(entry_ref)}"
            f" serve-vllm --port {shlex.quote(port)}\n",
            encoding="utf-8",
        )
    eval_cmd = workdir / "eval_cmd.txt"
    if not eval_cmd.is_file():
        extra = " ".join(
            shlex.quote(t) for t in _tokens(ctx.get("base_eval_args")))
        line = (
            "lm_eval --model local-chat-completions "
            '--tasks "${EVAL_TASKS}" '
            '--model_args "model=${SERVED_MODEL_NAME},base_url=${EVAL_BASE_URL},num_concurrent=16" '
            "--apply_chat_template --log_samples "
            '--output_path "${EVAL_OUTPUT_DIR}" '
            "${EVAL_LIMIT_ARG}")
        eval_cmd.write_text(f"{line} {extra}".rstrip() + "\n",
                            encoding="utf-8")
    return args


def summarize(workdir: Path, exit_code: int) -> dict[str, Any]:
    """Reduce the newest `eval_out/**/results_*.json` to `summary.json`."""
    ctx = _load_json_dict(workdir / CONTEXT_FILENAME, required=True)
    want = str(ctx.get("accuracy_metric") or "").strip()
    task = _default_eval_task(ctx)
    results = sorted((workdir / "eval_out").rglob("results_*.json"),
                     key=lambda p: p.stat().st_mtime)
    scores: dict[str, float] = {}
    if want and results:
        try:
            value = _extract_metric(results[-1], want, task)
        except ValueError:
            value = None
        if value is not None:
            scores[want] = value
    summary = {
        "status": "ok" if exit_code == 0 and scores else "failed",
        "exit_code": exit_code,
        "scores": scores,
    }
    (workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _is_multihost_worker(env: Mapping[str, str]) -> bool:
    try:
        num_hosts = int((env.get("TPU_NUM_HOSTS") or "1").strip())
    except ValueError:
        return False
    worker_id = (env.get("TPU_WORKER_ID") or env.get("JOB_COMPLETION_INDEX")
                 or "0").strip()
    return num_hosts > 1 and worker_id != "0"


def run_eval(workdir: Path, cli_args: Sequence[str]) -> int:
    """Run the shipped accuracy recipe, then write `summary.json` on host 0."""
    flags = _load_json_dict(workdir / FLAGS_FILENAME)
    env = _job_env(flags)
    args = eval_args(workdir, cli_args)
    proc = subprocess.run(
        ["bash", str(_launcher(workdir, "eval")), *args],
        cwd=workdir,
        env=env,
        check=False)
    if _is_multihost_worker(os.environ):
        return proc.returncode
    summary = summarize(workdir, proc.returncode)
    return 0 if summary["status"] == "ok" else (proc.returncode or 1)


def _pkg_version(name: str) -> str | None:
    if not name:
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_environment(
    ctx: Mapping[str, Any],
    env: Mapping[str, str],
    effective_env: Any = None,
) -> dict[str, str | None]:
    """Capture the 8–10 runtime environment fields inside the TPU job pod."""
    eff = {
        str(item["name"]): str(item["value"])
        for item in (effective_env if isinstance(effective_env, list) else ())
        if isinstance(item, Mapping) and "name" in item and "value" in item
    }
    engine = str(ctx.get("engine") or "").strip()
    env_doc: dict[str, str | None] = {
        "python_version":
        platform.python_version(),
        "jax_version":
        _pkg_version("jax"),
        "jaxlib_version":
        _pkg_version("jaxlib"),
        "torch_version":
        _pkg_version("torch"),
        "torch_tpu_version":
        _pkg_version("torch_tpu"),
        "engine_version":
        _pkg_version(engine),
        "libtpu_init_args": (eff.get("LIBTPU_INIT_ARGS")
                             or env.get("LIBTPU_INIT_ARGS") or None),
        "xla_flags": (eff.get("XLA_FLAGS") or env.get("XLA_FLAGS") or None),
    }
    raw_image = str(ctx.get("image") or "").strip()
    if raw_image:
        env_doc["image"] = raw_image
        _, sep, digest = raw_image.partition("@sha256:")
        if sep and digest:
            env_doc["image_digest"] = f"sha256:{digest}"
    return env_doc


def _point_metrics(rec: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    twin = rec.get("twin_model")
    if isinstance(twin, Mapping):
        for k, v in twin.items():
            fv = _finite_float(v)
            if fv is not None:
                out[str(k)] = fv
    if "output_tok_per_s" in out and "output_tokens_per_s" not in out:
        out["output_tokens_per_s"] = out["output_tok_per_s"]
    dec = rec.get("decode_ms")
    if isinstance(dec, Mapping):
        fv = _finite_float(dec.get("median_after_first"))
        if fv is not None:
            out.setdefault("decode_ms", fv)
    mixed = rec.get("mixed_ms")
    if isinstance(mixed, list) and mixed and isinstance(mixed[0], Mapping):
        fv = _finite_float(mixed[0].get("median_after_first"))
        if fv is not None:
            out.setdefault("mixed_ms", fv)
    return out


def _reduce_forward_result(
    result: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    points = result.get("points")
    if not isinstance(points, list):
        return []
    repeats = result.get("repeats")
    cells: list[dict[str, Any]] = []
    for idx, pt_rec in enumerate(points):
        if not isinstance(pt_rec, Mapping):
            continue
        pt = pt_rec.get("point") if isinstance(pt_rec.get("point"),
                                               Mapping) else {}
        prefill = pt.get("input")
        decode = pt.get("output")
        conc = pt.get("concurrency")
        cell_idx = idx
        for pc in protocol_cells:
            if (pc.get("prefill") == prefill and pc.get("decode") == decode
                    and pc.get("concurrency") == conc
                    and isinstance(pc.get("cell_index"), int)):
                cell_idx = int(pc["cell_index"])
                break
        reps: list[dict[str, float]] = []
        if isinstance(repeats, list) and repeats:
            for rep in repeats:
                if (isinstance(rep, list) and idx < len(rep)
                        and isinstance(rep[idx], Mapping)):
                    m = _point_metrics(rep[idx])
                    if m:
                        reps.append(m)
        elif (m := _point_metrics(pt_rec)):
            reps.append(m)
        if not reps:
            continue
        cell: dict[str, Any] = {
            "cell_index": cell_idx,
            "status": "ok",
            "replicates": reps,
        }
        if isinstance(prefill, int):
            cell["prefill"] = prefill
        if isinstance(decode, int):
            cell["decode"] = decode
        if isinstance(conc, int):
            cell["concurrency"] = conc
        cells.append(cell)
    return cells


def run_microbenchmark(workdir: Path, cli_args: Sequence[str]) -> int:
    """Run `forward_run.py` or a `.py` microbenchmark and emit canonical `summary.json`."""
    ctx = _load_json_dict(workdir / CONTEXT_FILENAME, required=True)
    flags = _load_json_dict(workdir / FLAGS_FILENAME)
    env = _job_env(flags)
    engine = str(ctx.get("engine") or "").strip()

    patch_file = workdir / PATCH_FILENAME
    if _has_nonempty_patch(patch_file):
        applier = workdir / "apply_vllm_patch.sh"
        if engine == "vllm" and applier.is_file():
            rc = subprocess.run(
                ["bash", str(applier), PATCH_FILENAME],
                cwd=workdir,
                env=env,
                check=False,
            ).returncode
            if rc != 0:
                return rc
        elif engine == "sglang":
            apply_patch(workdir)

    target = str(ctx.get("microbenchmark_target") or "").strip()
    if not target or not (workdir / target).is_file():
        raise FileNotFoundError(
            f"microbenchmark_target {target!r} missing in {workdir}")
    raw_pcells = ctx.get("protocol_cells")
    protocol_cells = ([c for c in raw_pcells if isinstance(c, Mapping)]
                      if isinstance(raw_pcells, list) else [])

    python = _resolve_python()
    if target.endswith(".json"):
        runner = workdir / "forward_run.py"
        if not runner.is_file():
            raise FileNotFoundError(f"forward_run.py missing in {workdir}")
        parsed_cli, _ = _parse_flags(cli_args, "--repeats", "--out")
        out_dir = Path(parsed_cli.out
                       or "wmb_out") if parsed_cli.out else workdir / "wmb_out"
        if not out_dir.is_absolute():
            out_dir = workdir / out_dir
        cmd = [python, str(runner), "--steps", str(workdir / target)]
        if parsed_cli.repeats is None:
            cmd += ["--repeats", str(int(ctx.get("min_replicates") or 1))]
        if parsed_cli.out is None:
            cmd += ["--out", str(out_dir)]
        cmd += [str(a) for a in cli_args]
        proc = subprocess.run(cmd, cwd=workdir, env=env, check=False)
        if _is_multihost_worker(os.environ):
            return proc.returncode
        res_doc = _load_json_dict(out_dir / "result.json")
        cells = _reduce_forward_result(res_doc, protocol_cells)
        env_doc = _collect_environment(ctx, env, res_doc.get("effective_env"))
        ok = proc.returncode == 0 and bool(cells)
        summary: dict[str, Any] = {
            "status": "ok" if ok else "failed",
            "exit_code": proc.returncode,
            "environment": env_doc,
            "cells": cells,
        }
    else:
        cmd = [python, str(workdir / target), *[str(a) for a in cli_args]]
        proc = subprocess.run(cmd, cwd=workdir, env=env, check=False)
        if _is_multihost_worker(os.environ):
            return proc.returncode
        raw_summary = _load_json_dict(workdir / "summary.json")
        raw_cells = raw_summary.get("cells")
        if isinstance(raw_cells, list):
            cells = raw_cells
        elif "replicates" in raw_summary:
            base_cell: dict[str, Any] = {
                "cell_index": 0,
                "status": "ok",
                "replicates": raw_summary["replicates"],
            }
            if len(protocol_cells) == 1:
                for k in ("prefill", "decode", "concurrency"):
                    if isinstance(protocol_cells[0].get(k), int):
                        base_cell[k] = protocol_cells[0][k]
            cells = [base_cell]
        else:
            cells = []
        env_doc = _collect_environment(ctx, env)
        if isinstance(raw_summary.get("environment"), Mapping):
            env_doc.update({
                str(k): v
                for k, v in raw_summary["environment"].items() if v is not None
            })
        ok = (proc.returncode == 0 and raw_summary.get("status", "ok") == "ok"
              and bool(cells))
        summary = {
            "status": "ok" if ok else "failed",
            "exit_code": proc.returncode,
            "environment": env_doc,
            "cells": cells,
        }

    (workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "ok" else (proc.returncode or 1)


_COMMANDS = ("serve-vllm", "serve-sglang", "eval-vllm", "run-microbenchmark")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch `serve-vllm`, `serve-sglang`, `eval-vllm`, or `run-microbenchmark`."""
    parser = argparse.ArgumentParser(
        description="Hillclimber target-side job runner")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="cmd", required=False)
    for name in _COMMANDS:
        sub.add_parser(name).add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    workdir = args.workdir.resolve()
    cmd = args.cmd
    extra = list(getattr(args, "extra", None) or [])
    if cmd is None:
        ctx = _load_json_dict(workdir / CONTEXT_FILENAME, required=True)
        cmd = str(ctx.get("runner_mode") or "").strip()
        if cmd not in _COMMANDS:
            raise ValueError(
                f"missing CLI subcommand and invalid runner_mode={cmd!r} in {CONTEXT_FILENAME}"
            )
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if cmd == "serve-vllm":
        return serve(workdir, "vllm", extra)
    if cmd == "serve-sglang":
        return serve(workdir, "sglang", extra)
    if cmd == "eval-vllm":
        return run_eval(workdir, extra)
    return run_microbenchmark(workdir, extra)


if __name__ == "__main__":
    raise SystemExit(main())
