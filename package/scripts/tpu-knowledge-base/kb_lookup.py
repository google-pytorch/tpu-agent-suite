#!/usr/bin/env python3
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
"""Provide relevant context for a task to be completed.

Look each key up in the catalog. A key matches a folder, a page on disk, a
page built earlier from a source, or a source that has not been built yet --
and the last two are the same thing at different ages. Keys that match
contribute their pages and references to one result; if none match, the
answer is a single prompt over all of them, and says it is grounded in
nothing.

A url the harness CLI could not summarise is downloaded to a local file and
comes back as a `to_read` item: the tpu-knowledge-base subagent summarises that
file with the instructions given, and keeps the summary with

    kb_lookup.py --keep ENTRY KEYWORD FILE [--run RUN_ID]

With `--no-crawl` (the request's `allow_crawl: false`) nothing is read now:
only pages on disk and current kept pages are served, and a url with
neither is cited in `not_read`.
"""

import argparse
import importlib.util
import json
import pathlib
import re
import shlex
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog = _load("catalog")
rank = _load("rank")
summarize = _load("kb_summarize")


def served_path(query: str) -> pathlib.Path | None:
    """A page or folder the caller named by path rather than by keyword.

    A caller that read an answer cites its path, so the path has to be a way
    back in; without this it falls through to ranked search and lands on
    whatever shares the most words with it.
    """
    if "/" not in query:
        return None
    root = catalog.PACKAGE_DIR.resolve()
    candidate = (root / query).resolve()
    if not candidate.is_relative_to(root) or not candidate.exists():
        return None
    return candidate if candidate.is_dir(
    ) or candidate.suffix == ".md" else None


SOURCE_SECTION = "<!-- sources:generated:start -->"

HEADING = re.compile(r"^## (?:Sources|References)\s*$", re.M)

# What a summary build can fail with. The harness CLI timing out is one: let
# through, one slow page ended the whole lookup and every other key's result
# with it.
BUILD_FAILURES = (ValueError, OSError, summarize.subprocess.TimeoutExpired)

URL = re.compile(r"https?://[^\s)\"'>\]]+")

BY_KEYWORD = re.compile(r"index\.py get ([a-z0-9][a-z0-9._-]*)")


def cited_section(text: str) -> str:
    """The part of a page that names its sources, empty if it names none.

    Scoped rather than whole-page: a URL in prose is an aside, and citing it
    would footnote a claim it does not support.
    """
    if SOURCE_SECTION in text:
        return text.split(SOURCE_SECTION, 1)[1]
    heading = HEADING.search(text)
    return text[heading.end():] if heading else ""


def citations(
    page: pathlib.Path,
    entry: dict | None,
    entries: list[dict],
    seen: frozenset[pathlib.Path] = frozenset()
) -> list[dict]:
    """Footnote targets, preferring what was recorded over what was written.

    A generated page keeps its evidence in a sibling record, so that is read
    first. Failing that the page is asked: a built one names its url in the
    header, a shipped one in its sources section -- where a legacy page
    cites keywords rather than links, and those resolve through the catalog.

    `authority` comes from the index row, and only for a url that row
    actually asserts. A url scraped out of prose is one the page's author
    wrote down -- possibly a model -- so it is `inferred`, and a caller can
    tell a vouched-for citation from one it must check itself.
    """
    authority = (entry or {}).get("authority", "inferred")
    asserted = {(entry or {}).get("url")} - {None, ""}
    record = page.with_suffix(".generation.json")
    if record.is_file():
        try:
            fetched = json.loads(record.read_text("utf-8"))["input_sources"]
            return [{
                "url": row["url"],
                "authority": authority
            } for row in fetched if row.get("url")]
        except (OSError, ValueError, KeyError):
            pass
    try:
        text = page.read_text("utf-8", errors="replace")
    except OSError:
        # An answer that exists is worth more than its footnotes; the caller
        # sees an empty list and knows not to cite.
        return []
    # A summary of a page on disk cites what that page cited: the row
    # has no url of its own, so without this the references are lost.
    # `origin` is read out of a page a model wrote, so it is untrusted:
    # follow it only inside the package or the cache, and only once.
    origin = catalog.front_field(text, "origin")
    if origin:
        try:
            where = pathlib.Path(origin).resolve()
        except OSError:
            where = None
        roots = (catalog.PACKAGE_DIR.resolve(),
                 summarize.CACHE_DIRS[0].resolve())
        if (where and where.is_file() and where not in seen
                and any(where.is_relative_to(r) for r in roots)):
            return citations(where, entry, entries, seen | {where})
    urls: dict[str, str] = {}
    for url in URL.findall(cited_section(text)):
        urls.setdefault(url, "inferred")
    # A keyword resolves to a row, so that row vouches for its own url.
    for keyword in BY_KEYWORD.findall(cited_section(text)):
        found = catalog.find(entries, keyword)
        if found and found[0].get("url"):
            urls[found[0]["url"]] = found[0].get("authority", "inferred")
    header = catalog.front_field(text, "url")
    # An authored page with no source says so in this field
    # ("not applicable -- see Evidence"); that is not a url to cite.
    if header and URL.fullmatch(header):
        urls[header] = authority if header in asserted else "inferred"
    for url in asserted:
        urls[url] = authority
    return [{"url": u, "authority": a} for u, a in urls.items()]


def match(
        entries: list[dict],
        key: str,
        harness_name: str | None,
        run_id: str = "",
        unread: list[dict] | None = None,
        not_read: list[dict] | None = None) -> tuple[str | None, dict | None]:
    """What a key points at: a page on disk, or the url of one.

    Nothing is fetched or built here. A source is a thing to read, and the
    reader decides how -- which is why a url needs no special case.
    """
    direct = served_path(key)
    if direct:
        return str(direct), None
    found = catalog.find(entries, key)
    if not found:
        return None, None
    return serve(*found, harness_name, run_id, unread, not_read)


def serve(
        row: dict,
        keyword: str,
        harness_name: str | None,
        run_id: str = "",
        unread: list[dict] | None = None,
        not_read: list[dict] | None = None) -> tuple[str | None, dict | None]:
    """The thing to read for a row, however the row was arrived at.

    A url the CLI could not summarise is added to `unread`, so the caller
    can read it itself rather than take the bare url for a summary.

    `not_read` given is `--no-crawl`: a url with no current kept page is
    added to it and nothing reads it -- no CLI build, no download.
    """
    page = catalog.page_of(row)
    source = str(page) if page else row.get("url", "").replace(
        "{keyword}", keyword)
    if not source:
        return None, row
    if not summarize.worth_keeping(source):
        return source, row
    # Expensive to read again: serve the summary if one is current --
    # dated within the TTL, built from the url the row names today and
    # read with the instructions it names today -- and make one if not. A
    # build that fails hands back the source it could not summarise, so
    # the reader can still try.
    #
    # The identity is the kind's template, not the text the model sees:
    # every alias of a row shares one page, and the alias is in the text.
    # A prompt table that cannot be read hands back the url the same way
    # a failed build does -- not an empty template, which would build and
    # keep a page read with no instructions at all.
    try:
        template = catalog.instructions_template(row.get("prompt"))
    except (ValueError, OSError):
        return source, row
    kept = summarize.cache_file(row, keyword)
    if kept.is_file() and not summarize.expired(kept, row, template):
        return str(kept), row
    if not_read is not None:
        not_read.append({
            "source": source,
            "entry": row.get("id", ""),
            "keyword": keyword
        })
        return source, row
    try:
        return str(
            summarize.build(row, keyword, source, template, harness_name,
                            run_id)), row
    except BUILD_FAILURES as exc:
        if unread is not None:
            unread.append(
                to_read(row, keyword, source, template, run_id, str(exc)))
        return source, row


def downloaded(row: dict, keyword: str, source: str) -> tuple[str | None, str]:
    """A local copy of `source` for the subagent to summarise, or why not."""
    try:
        return str(summarize.fetch(row, keyword, source)), ""
    except summarize.FETCH_ERRORS as exc:
        return None, str(exc) or type(exc).__name__


def fallback(item: dict) -> str:
    """One line saying what failed for a source and what was done instead.

    Said in the result and on stderr, so a trajectory shows that a fallback
    was taken and why, whichever of the two its harness records.
    """
    if item["file"]:
        then = (f"downloaded it to {item['file']}: summarise that file and "
                f"run its `keep` command")
    else:
        then = (f"the download failed too ({item['fetch_error']}): read it "
                f"with your own access, or say it could not be read")
    return (f"fallback: the harness CLI could not summarise "
            f"{item['source']} ({item['failed']}); {then}.")


def to_read(row: dict,
            keyword: str,
            source: str,
            template: str,
            run_id: str = "",
            failed: str = "") -> dict:
    """What the caller needs to summarise `source` itself and keep it.

    `failed` is why the CLI could not: it used to be swallowed, and a bare
    url in `sources` read as if it had been summarised. `file` is the page,
    downloaded here so the caller reads it once and the same way every
    time; `fetch_error` is why it is null.
    """
    file, fetch_error = downloaded(row, keyword, source)
    keep = [
        "python3",
        str(pathlib.Path(__file__).resolve()), "--keep",
        row.get("id", ""), keyword, "<file>"
    ]
    if run_id:
        keep += ["--run", run_id]
    instructions = (
        summarize.AS_DATA + catalog.with_keyword(template, keyword) +
        summarize.asking_for(summarize.critical_fields(row, keyword)))
    return {
        "source": source,
        "file": file,
        "fetch_error": fetch_error,
        "entry": row.get("id", ""),
        "keyword": keyword,
        "instructions": instructions,
        # `<file>` is where the caller saved its summary; the rest is
        # quoted so the line can be pasted once that is filled in.
        "keep": " ".join(shlex.quote(a) if a != "<file>" else a for a in keep),
        "failed": failed,
    }


# A key names a catalog entry, so it is short and single-line. Anything
# else is not a key, and the fallback puts it in a prompt.
MAX_KEY = 200

# `<` and `>` for the same reason as the newline: the fallback wraps the
# keys in a `<question>` tag, and a key carrying one closes it early and
# writes the rest of itself outside the span marked as data.
NOT_IN_A_KEY = "\n\r\x00<>"


def sane(key: str) -> bool:
    """Whether this is a key at all, rather than prose or a payload."""
    return bool(key and key.strip() and len(key) <= MAX_KEY
                and not any(c in key for c in NOT_IN_A_KEY))


# How many near misses to hand back. Five is what the ranking is good
# for: on the 566 labelled queries in tests/tpu-knowledge-base, the entry a
# caller meant is in the top five 89% of the time -- and is first only 67%
# of the time, which is why these are offered rather than served.
SUGGESTIONS = 5

# The score below which a row is not offered at all. Without one, almost
# any words came back `near`: "what is the capital of france" offered
# vllm-gpt-oss-mxfp4 at 4.3. At 6, one of the 566 labelled queries loses
# its answer (CompileTimeScopedVmemOom, found at 0.75), and the same one
# of the 516 mistyped. Off-topic words still clear it -- "kubernetes pod
# eviction" scores 9.2 -- which is why the caller is told to choose.
MIN_SCORE = 6.0


def suggest(entries: list[dict], key: str) -> list[dict]:
    """Rows this key might have meant, best first.

    Not served, and deliberately. Top-1 is right 67% of the time, and a
    wrong one costs more than a turn: a url row would be summarised by the
    harness before anyone could see it was the wrong page. Scores do not
    separate the right guess from the wrong one either -- correct and
    incorrect top hits have close medians, 21.7 against 18.5, with the two
    spreads overlapping, so no threshold picks the right one; MIN_SCORE
    only drops what nothing real scores as low as. What does separate them
    is the question itself, which the caller has and the ranking does not.

    A row with no keyword is not offered: there is nothing to ask again
    with. The templated row is the one, and a chip name reaches it anyway.
    """
    hits = rank.search(entries, key, catalog.page_of, len(entries),
                       catalog.page_scored)
    return [{
        "keyword": keyword,
        "entry": row.get("id", ""),
        "summary": row.get("summary", ""),
        "score": round(score, 2),
    } for score, row, keyword in hits
            if keyword and score >= MIN_SCORE][:SUGGESTIONS]


def respelling(entries: list[dict], key: str) -> str:
    """` (read as 'sharding')` when the ranking respelt `shardng`, else "".

    A respelling is a guess like any other -- `mistral` becomes `mixtral`,
    a different model -- so the caller is told which words were swapped.
    """
    swaps = rank.respellings(entries, key, catalog.page_of,
                             catalog.page_scored)
    if not swaps:
        return ""
    return " (read as " + ", ".join(repr(new) for _, new in swaps) + ")"


def route(entries: list[dict],
          keys: list[str],
          harness_name: str | None,
          run_id: str = "",
          crawl: bool = True) -> dict:
    """Answer every key the catalog serves, as one result.

    `partial` is `cached` with `to_read` items left: the result is not
    complete until the caller summarises and keeps them.

    `crawl` off is the request's `allow_crawl: false`: nothing is read now.
    A url with no current kept page is cited, in `references` and
    `not_read`, and is not a source; a key nothing serves gets no answer
    from the harness. So there is never a `to_read` item, nor `partial`.

    Three tiers, in order. A key that names a keyword is served exactly. A
    key that names none is ranked, and the best rows offered, not served --
    "why is my flash attn slow" is no keyword and is still about
    flash-attention-to-sdpa. Keys that reach neither are asked as one
    question with nothing behind it.
    """
    sources: list[str] = []
    references: list[dict] = []
    matched: list[str] = []
    near: dict[str, list[dict]] = {}
    read_as: dict[str, str] = {}
    unread: list[dict] = []
    not_read: list[dict] | None = None if crawl else []
    for key in dict.fromkeys(k for k in keys if sane(k)):
        cited_before = len(not_read or [])
        source, row = match(entries, key, harness_name, run_id, unread,
                            not_read)
        if source is None:
            offered = suggest(entries, key)
            if offered:
                near[key] = offered
                read_as[key] = respelling(entries, key)
            continue
        matched.append(key)
        if not_read and len(not_read) > cited_before:
            # A citation without content: the caller may name the url, and
            # nothing on it is established.
            references.append({
                "url":
                source,
                "authority": (row or {}).get("authority", "inferred")
            })
            continue
        path = pathlib.Path(source)
        pages = ([str(p) for p in sorted(path.rglob("*.md"))]
                 if path.is_dir() else [source])
        sources.extend(pages)
        for page in pages:
            references.extend(
                citations(pathlib.Path(page), row, entries) if pathlib.
                Path(page).is_file(
                ) else [{
                    "url": page,
                    "authority": (row or {}).get("authority", "inferred")
                }])

    unmatched = [k for k in dict.fromkeys(keys) if k and k not in matched]
    if sources:
        aside = "".join(
            f" {key!r}{read_as[key]} named no keyword; did you "
            f"mean {', '.join(repr(s['keyword']) for s in offered)}?"
            for key, offered in near.items())
        reading = list({u["source"]: u for u in unread}.values())
        not_summarised = "".join(f" {fallback(u)}" for u in reading)
        for item in reading:
            print(f"tpu-knowledge-base {fallback(item)}", file=sys.stderr)
        # `partial` until every url has a summary: under `cached`, the
        # subagent read the local pages, took the result as complete and
        # never opened a download. The next lookup after `--keep` serves
        # the kept pages, and is `cached` again.
        status = "partial" if reading else "cached"
        lead = (f"partial: {len(reading)} of {len(sources)} source(s) still "
                f"need your summary; do each `to_read` item before you "
                f"answer. " if reading else "")
        found = {
            "status":
            status,
            "summary": (f"{lead}{len(sources)} source(s) for "
                        f"{', '.join(matched)}; read them at `sources`."
                        f"{not_summarised}{aside}"),
            "sources":
            sources,
            "references":
            list({r["url"]: r
                  for r in references}.values()),
            "matched":
            matched,
            "suggestions":
            near,
            "unmatched":
            unmatched,
        }
        if reading:
            found["to_read"] = reading
        if not_read:
            found["summary"] += uncrawled_note(not_read)
            found["not_read"] = list({n["source"]: n
                                      for n in not_read}.values())
        return found

    if not_read:
        # Every key was served, and none of them with content.
        return {
            "status":
            "cited-only",
            "summary": (f"{', '.join(matched)}: nothing kept to serve."
                        f"{uncrawled_note(not_read)}"),
            "sources": [],
            "references":
            list({r["url"]: r
                  for r in references}.values()),
            "matched":
            matched,
            "suggestions":
            near,
            "unmatched":
            unmatched,
            "not_read":
            list({n["source"]: n
                  for n in not_read}.values()),
        }

    # No key was served. One prompt over all of them, written and marked so
    # it is never mistaken for something the catalog answered. The harness
    # runs on its own defaults, so it may well have looked something up --
    # what this says is only that no source here vouches for the answer.
    if near:
        # Nothing resolved, but the ranking knows what these are about. A
        # list the caller chooses from beats an answer nothing vouches
        # for, and it costs no model call to produce.
        return {
            "status":
            "near",
            "summary": ("no key named a catalog entry. " +
                        " ".join(f"For {key!r}{read_as[key]}: " +
                                 ", ".join(f"{s['keyword']} ({s['entry']})"
                                           for s in offered) + "."
                                 for key, offered in near.items()) +
                        " Nothing was looked up: ask again with whichever "
                        "fits the question, if one does."),
            "sources": [],
            "references": [],
            "matched": [],
            "suggestions":
            near,
            "unmatched":
            unmatched,
        }

    asked = ", ".join(k for k in keys if sane(k))
    if not asked:
        # Nothing here is a key, so there is nothing to look up and
        # nothing worth putting in a prompt.
        return {
            "status": "uncited",
            "summary": "no key given that names a catalog entry.",
            "sources": [],
            "references": [],
            "matched": [],
            "suggestions": {},
            "unmatched": unmatched,
        }
    if not crawl:
        # Answering without a source is a harness call like any other read.
        return {
            "status":
            "uncited",
            "summary": ("no catalog page serves these keys, and crawling "
                        "was off, so nothing answered them: they are not "
                        "established."),
            "sources": [],
            "references": [],
            "matched": [],
            "suggestions": {},
            "unmatched":
            unmatched,
        }
    error = None
    try:
        sources = [str(summarize.unsourced(asked, harness_name, run_id))]
        summary = ("no catalog page serves these keys; the answer is the "
                   "harness's own and nothing here vouches for it.")
    except BUILD_FAILURES as exc:
        # A miss is an answer, so this still exits 0: a traceback sent the
        # caller looking through the package by hand. But a failed answer
        # is not an empty catalog either, and with the error only in the
        # summary a caller took it for one and went looking anyway. The
        # `error` field is what tells the two apart.
        sources = []
        error = f"the harness CLI could not answer without a source: {exc}"
        summary = (f"no catalog page serves these keys, and the lookup "
                   f"failed answering without one: {exc}. The tool failed; "
                   f"the catalog was searched.")
    return {
        "status": "uncited",
        "summary": summary,
        "sources": sources,
        "references": [],
        "matched": [],
        "suggestions": {},
        "unmatched": unmatched,
        **({
            "error": error
        } if error else {}),
    }


def uncrawled_note(not_read: list[dict]) -> str:
    """What a lookup with crawling off left out, for the summary line."""
    entries = ", ".join(dict.fromkeys(n["entry"] for n in not_read))
    return (f" {len(not_read)} url(s) cited but not read, because crawling "
            f"was off: {entries}. Nothing on them is established.")


def keep_cli(entry: str, keyword: str, path: str, run_id: str) -> dict:
    """`--keep`: cache what the tpu-knowledge-base subagent read for `entry`."""
    row = next((r for r in catalog.rows() if r.get("id") == entry), None)
    if row is None:
        raise ValueError(f"no catalog entry {entry!r}")
    template = catalog.instructions_template(row.get("prompt"))
    body = pathlib.Path(path).read_text("utf-8")
    kept = summarize.keep(row, keyword, body, template, run_id)
    # Read and summarised: the download has done its job.
    summarize.fetched_file(row, keyword).unlink(missing_ok=True)
    return {
        "status":
        "kept",
        "entry":
        entry,
        "keyword":
        keyword,
        "path":
        str(kept),
        "next": ("run the lookup again: once every `to_read` item is kept "
                 "it serves the kept pages and returns `cached`."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a knowledge request.")
    parser.add_argument("keys", nargs="*", help="Keywords, or a page path")
    parser.add_argument("--run",
                        dest="run_id",
                        default="",
                        help="Session run id. Recorded on anything built, "
                        "and the hook for session state later")
    parser.add_argument("--harness",
                        choices=sorted(summarize.HARNESS_CLI),
                        help="Harness CLI to answer with; probed by default")
    parser.add_argument("--no-crawl",
                        dest="crawl",
                        action="store_false",
                        help="Serve only what is already readable: a url "
                        "with no current kept page is cited, not built or "
                        "downloaded (the request's allow_crawl: false)")
    parser.add_argument("--keep",
                        nargs=3,
                        metavar=("ENTRY", "KEYWORD", "FILE"),
                        help="Cache the summary in FILE that you wrote from "
                        "a `to_read` item, under that item's entry and "
                        "keyword")
    args = parser.parse_args(argv)
    if args.keep:
        if args.keys:
            parser.error("--keep takes no keys")
        try:
            kept = keep_cli(*args.keep, args.run_id)
        except (ValueError, OSError) as exc:
            # Nothing was kept. Said in the same shape as a result, and
            # non-zero, because the caller asked for something to happen.
            print(
                json.dumps({
                    "status": "not-kept",
                    "reason": str(exc)
                },
                           indent=2))
            return 1
        print(json.dumps(kept, indent=2))
        return 0
    if not args.keys:
        parser.error("give at least one key, or --keep ENTRY KEYWORD FILE")
    found = route(catalog.rows(), args.keys, args.harness, args.run_id,
                  args.crawl)
    print(json.dumps(found, indent=2))
    # Zero whenever a record was produced. Exiting non-zero on `uncited`
    # told a caller its tool had failed, and a caller that believes that
    # goes looking through the package by hand.
    return 0


if __name__ == "__main__":
    sys.exit(main())
