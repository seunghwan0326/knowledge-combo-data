#!/usr/bin/env python3
"""Generate 2,700 Knowledge Combo questions from verified atomic facts already in repo.

36 topics x 5 difficulties x 15 questions = 2,700.
No external model/API. New questions are new combinations of previously verified atomic Q/A facts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOPICS = [
    ("general", "일반상식"), ("capital", "세계 수도"), ("world_geo", "세계 지리"), ("korea_geo", "한국 지리"),
    ("korean_history", "한국사"), ("world_history", "세계사"), ("science", "과학 상식"), ("astronomy", "우주·천문"),
    ("biology", "동물·식물"), ("health", "인체·건강"), ("environment", "환경·기후"), ("korean_language", "국어·한글"),
    ("literature", "문학·작가"), ("art_music", "미술·음악"), ("sports", "스포츠"), ("food_culture", "음식·문화"),
    ("technology", "발명·기술"), ("economy", "경제·사회"), ("people", "인물 맞히기"), ("proverbs", "속담·사자성어"),
    ("bible", "성경"), ("elementary_korean", "초등 국어"), ("elementary_math", "초등 수학"), ("elementary_social", "초등 사회"),
    ("elementary_science", "초등 과학"), ("elementary_english", "초등 영어"), ("middle_korean", "중등 국어"), ("middle_math", "중등 수학"),
    ("middle_social", "중등 사회"), ("middle_science", "중등 과학"), ("middle_english", "중등 영어"), ("high_korean", "고등 국어"),
    ("high_math", "고등 수학"), ("high_social", "고등 사회"), ("high_science", "고등 과학"), ("high_english", "고등 영어"),
]
TOPIC_NAME = dict(TOPICS)
DIFFICULTIES = ["easy", "normal", "hard", "expert", "master"]
DIFF_LABEL = {"easy": "하", "normal": "중", "hard": "상", "expert": "최상", "master": "최최상"}
FACT_COUNT = {"easy": 2, "normal": 2, "hard": 3, "expert": 4, "master": 4}
ADJACENT = {
    "easy": ["easy", "normal"],
    "normal": ["normal", "easy", "hard"],
    "hard": ["hard", "normal", "expert"],
    "expert": ["expert", "hard", "master"],
    "master": ["master", "expert"],
}
MARKERS = ["가", "나", "다", "라", "마", "바"]
MARKER_START_RE = re.compile(r"\n\s*\(가\)\s*")
SUBQ_RE = re.compile(
    r"\((가|나|다|라|마|바)\)\s*(.*?)(?=\n\s*\((?:가|나|다|라|마|바)\)\s*|\Z)",
    re.S,
)
PACK_SEQ_RE = re.compile(r"daily-knowledge-combo-2700-\d{8}-(\d+)$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def split_subquestions(question: str) -> list[str]:
    m = MARKER_START_RE.search(question)
    if not m:
        return []
    tail = question[m.start() + 1 :]
    return [x.group(2).strip() for x in SUBQ_RE.finditer(tail)]


def correct_answer_text(q: dict[str, Any]) -> str | None:
    ans = q.get("answer")
    if isinstance(ans, str) and ans.strip():
        return ans.strip()
    choices = q.get("choices")
    if isinstance(ans, int) and isinstance(choices, list) and choices:
        if 0 <= ans < len(choices):
            return str(choices[ans]).strip()
        if 1 <= ans <= len(choices):
            return str(choices[ans - 1]).strip()
    return None


def extract_atoms(q: dict[str, Any]) -> list[dict[str, str]]:
    topic = str(q.get("topic", "")).strip()
    difficulty = str(q.get("difficulty", "")).strip()
    if topic not in TOPIC_NAME or difficulty not in DIFFICULTIES:
        return []
    question = str(q.get("question", "")).strip()
    answer = correct_answer_text(q)
    if not question or not answer:
        return []

    subs = split_subquestions(question)
    if subs:
        answers = [x.strip() for x in re.split(r"\s+/\s+", answer)]
        src = q.get("sourceQuestionIds") or []
        if len(subs) == len(answers) and all(answers):
            out = []
            for i, (prompt, a) in enumerate(zip(subs, answers)):
                # Slash inside an atomic answer would make future composite parsing ambiguous.
                if "/" in a:
                    continue
                sid = (
                    str(src[i])
                    if isinstance(src, list) and len(src) == len(subs)
                    else f"atom_{stable_hash(topic, prompt, a)[:20]}"
                )
                out.append({"topic": topic, "difficulty": difficulty, "prompt": prompt, "answer": a, "source_id": sid})
            if out:
                return out

    if "/" in answer:
        return []
    sid = str(q.get("id") or f"atom_{stable_hash(topic, question, answer)[:20]}")
    return [{"topic": topic, "difficulty": difficulty, "prompt": question, "answer": answer, "source_id": sid}]


def source_paths(root: Path) -> list[Path]:
    """Candidate truth sources: non-cumulative, non-validation JSON files.

    Dated 2700 packs are *not* used as truth sources because a bad composite
    must never become the next day's atom. They are read separately only for
    duplicate source-combination history.
    """
    out = []
    for p in root.glob("*.json"):
        n = p.name.lower()
        if n in {"manifest.json", "manifest-1.json", "manifest-2.json", "question_pack_template.json", "auto_validation_report.json"}:
            continue
        if "validation" in n or "backup" in n:
            continue
        if n.startswith("daily_knowledge_combo_"):
            continue
        out.append(p)
    return sorted(out)


def history_paths(root: Path) -> list[Path]:
    return sorted(root.glob("daily_knowledge_combo_*.json"))


def single_atom_consistent(q: dict[str, Any], atom: dict[str, str]) -> bool:
    answer = atom["answer"]
    choices = q.get("choices")
    if isinstance(choices, list) and choices:
        if not any(norm(c) == norm(answer) for c in choices):
            return False
    accepted = q.get("accepted")
    if isinstance(accepted, list) and accepted:
        if not any(norm(a) == norm(answer) for a in accepted):
            return False
    explanation = str(q.get("explanation", "")).strip()
    # Knowledge source packs normally echo the verified answer in explanation.
    # Requiring that support blocks internally inconsistent records.
    if explanation and norm(answer) not in norm(explanation):
        return False
    return True


def build_sources(root: Path):
    candidates: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    historical_signatures: set[tuple[str, ...]] = set()
    historical_questions: set[str] = set()

    # Only original single questions become atom truth sources.
    for path in source_paths(root):
        try:
            data = load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
            continue
        for q in data["questions"]:
            if not isinstance(q, dict):
                continue
            question = str(q.get("question", ""))
            if split_subquestions(question):
                continue
            atoms = extract_atoms(q)
            if len(atoms) != 1:
                continue
            atom = atoms[0]
            if not single_atom_consistent(q, atom):
                continue
            key = (atom["topic"], atom["difficulty"], norm(atom["prompt"]))
            candidates[key].append(atom)

    pools: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for (topic, difficulty, _), variants in candidates.items():
        answers = {norm(v["answer"]) for v in variants}
        # Same prompt with conflicting answers is quarantined.
        if len(answers) != 1:
            continue
        variants.sort(key=lambda v: (len(v["source_id"]), v["source_id"]))
        pools[(topic, difficulty)].append(variants[0])

    # Dated composites contribute only "already used" signatures/questions.
    for path in history_paths(root):
        try:
            data = load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
            continue
        for q in data["questions"]:
            if not isinstance(q, dict):
                continue
            nq = norm(q.get("question"))
            if nq:
                historical_questions.add(nq)
            src = q.get("sourceQuestionIds")
            if isinstance(src, list) and len(src) >= 2:
                historical_signatures.add(tuple(sorted(map(str, src))))

    return dict(pools), historical_signatures, historical_questions

def candidate_pool(pools, topic: str, difficulty: str, broaden: bool) -> list[dict[str, str]]:
    diffs = ADJACENT[difficulty] if broaden else [difficulty]
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for d in diffs:
        for a in pools.get((topic, d), []):
            merged.setdefault((norm(a["prompt"]), norm(a["answer"])), a)
    # Source ID should be unique in a combination.
    by_source: dict[str, dict[str, str]] = {}
    for a in merged.values():
        by_source.setdefault(a["source_id"], a)
    return list(by_source.values())


def select_unused_combos(
    atoms: list[dict[str, str]], n_facts: int, needed: int, date: str, salt: str,
    blocked: set[tuple[str, ...]], selected_global: set[tuple[str, ...]],
) -> list[tuple[dict[str, str], ...]]:
    if len(atoms) < n_facts:
        return []
    total = math.comb(len(atoms), n_facts)
    candidates: list[tuple[dict[str, str], ...]] = []

    def valid(combo):
        sig = tuple(sorted(a["source_id"] for a in combo))
        if sig in blocked or sig in selected_global:
            return False
        if len({norm(a["prompt"]) for a in combo}) != n_facts:
            return False
        if len({norm(a["answer"]) for a in combo}) < max(2, n_facts - 1):
            return False
        return True

    if total <= 250_000:
        for combo in itertools.combinations(atoms, n_facts):
            if valid(combo):
                candidates.append(combo)
        candidates.sort(key=lambda c: stable_hash(date, salt, *(a["source_id"] for a in c)))
        return candidates[:needed]

    rng = random.Random(int(stable_hash(date, salt)[:16], 16))
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(candidates) < needed and attempts < 50_000:
        attempts += 1
        combo = tuple(rng.sample(atoms, n_facts))
        sig = tuple(sorted(a["source_id"] for a in combo))
        if sig in seen:
            continue
        seen.add(sig)
        if valid(combo):
            candidates.append(combo)
    return candidates


def distractors(correct_parts: list[str], atom_pool: list[dict[str, str]], rng: random.Random) -> list[str]:
    correct = " / ".join(correct_parts)
    alternatives = list(dict.fromkeys(a["answer"] for a in atom_pool if "/" not in a["answer"] and norm(a["answer"]) not in {norm(x) for x in correct_parts}))
    if len(alternatives) < 3:
        raise RuntimeError("Not enough distinct answer alternatives")
    out = []
    attempts = 0
    while len(out) < 3 and attempts < 1000:
        attempts += 1
        parts = list(correct_parts)
        change_count = rng.choice([1, 1, 2]) if len(parts) >= 3 else 1
        for pos in rng.sample(range(len(parts)), k=min(change_count, len(parts))):
            parts[pos] = rng.choice(alternatives)
        c = " / ".join(parts)
        if norm(c) != norm(correct) and c not in out:
            out.append(c)
    if len(out) != 3:
        raise RuntimeError("Could not make 3 unique distractors")
    return out


def next_pack_id(manifest: dict[str, Any], target_date: str) -> str:
    compact = target_date.replace("-", "")
    for p in manifest.get("packs", []):
        if isinstance(p, dict) and str(p.get("url", "")) == f"daily_knowledge_combo_{compact}_2700.json":
            return str(p.get("id"))
    max_seq = 0
    for p in manifest.get("packs", []):
        if not isinstance(p, dict):
            continue
        m = PACK_SEQ_RE.match(str(p.get("id", "")))
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return f"daily-knowledge-combo-2700-{compact}-{max_seq + 1:03d}"


def generate(root: Path, target_date: str, manifest: dict[str, Any]) -> dict[str, Any]:
    compact = target_date.replace("-", "")
    pack_id = next_pack_id(manifest, target_date)
    pools, historical_signatures, historical_questions = build_sources(root)
    selected_global: set[tuple[str, ...]] = set()
    questions: list[dict[str, Any]] = []

    for topic, topic_name in TOPICS:
        for difficulty in DIFFICULTIES:
            n_facts = FACT_COUNT[difficulty]
            chosen: list[tuple[dict[str, str], ...]] = []
            for broaden in (False, True):
                pool = candidate_pool(pools, topic, difficulty, broaden)
                need = 15 - len(chosen)
                if need <= 0:
                    break
                more = select_unused_combos(
                    pool, n_facts, need, target_date,
                    f"{topic}:{difficulty}:{'broad' if broaden else 'exact'}",
                    historical_signatures, selected_global,
                )
                for combo in more:
                    sig = tuple(sorted(a["source_id"] for a in combo))
                    if sig not in selected_global:
                        chosen.append(combo)
                        selected_global.add(sig)
                if len(chosen) >= 15:
                    break
            if len(chosen) != 15:
                exact_n = len(candidate_pool(pools, topic, difficulty, False))
                broad_n = len(candidate_pool(pools, topic, difficulty, True))
                raise RuntimeError(f"{topic}/{difficulty}: only {len(chosen)}/15 new combos; atom pools exact={exact_n}, broad={broad_n}")

            answer_pool = candidate_pool(pools, topic, difficulty, True)
            for idx, combo in enumerate(chosen, 1):
                parts = [a["answer"] for a in combo]
                correct = " / ".join(parts)
                rng = random.Random(int(stable_hash(target_date, topic, difficulty, str(idx))[:16], 16))
                choices = distractors(parts, answer_pool, rng) + [correct]
                rng.shuffle(choices)
                blocks = [f"({MARKERS[i]}) {a['prompt']}" for i, a in enumerate(combo)]
                question = (
                    f"[{topic_name}·{DIFF_LABEL[difficulty]}] 다음 {len(combo)}개 물음의 답을 "
                    f"(가)부터 순서대로 바르게 짝지은 것은?\n\n" + "\n".join(blocks)
                )
                if norm(question) in historical_questions:
                    raise RuntimeError(f"Exact historical duplicate detected: {topic}/{difficulty}/{idx}")
                historical_questions.add(norm(question))
                explanation = "정답 대응은 " + ", ".join(f"({MARKERS[i]}) {a['answer']}" for i, a in enumerate(combo)) + "입니다."
                questions.append(
                    {
                        "id": f"kc_{compact}_{topic}_{difficulty}_{idx:02d}",
                        "topic": topic,
                        "topicName": topic_name,
                        "difficulty": difficulty,
                        "question": question,
                        "answer": correct,
                        "accepted": [correct],
                        "choices": choices,
                        "hint": "각 물음을 따로 해결한 뒤 답의 순서를 확인하세요.",
                        "explanation": explanation,
                        "date": target_date,
                        "packId": pack_id,
                        "sourceQuestionIds": [a["source_id"] for a in combo],
                        "questionMode": "multi_fact_composite",
                    }
                )

    previous = int(manifest.get("cumulativeQuestionCount", 0))
    return {
        "schemaVersion": 1,
        "packId": pack_id,
        "date": target_date,
        "title": f"{target_date} 지식콤보 2,700문제",
        "questionCount": 2700,
        "previousCumulativeQuestionCount": previous,
        "cumulativeQuestionCount": previous + 2700,
        "questions": questions,
    }


def update_manifest(manifest: dict[str, Any], pack: dict[str, Any], target_date: str) -> dict[str, Any]:
    compact = target_date.replace("-", "")
    url = f"daily_knowledge_combo_{compact}_2700.json"
    packs = manifest.setdefault("packs", [])
    existing = None
    for p in packs:
        if isinstance(p, dict) and p.get("url") == url:
            existing = p
            break

    if existing is None:
        packs.append({"id": pack["packId"], "version": 1, "url": url, "enabled": True, "questionCount": 2700})
        manifest["cumulativeQuestionCount"] = int(pack["cumulativeQuestionCount"])
    else:
        # Same-day idempotent repair: do not add another 2,700.
        existing.update({"id": pack["packId"], "version": existing.get("version", 1), "url": url, "enabled": True, "questionCount": 2700})
        manifest["cumulativeQuestionCount"] = max(int(manifest.get("cumulativeQuestionCount", 0)), int(pack.get("cumulativeQuestionCount", 0)))

    manifest["updatedAt"] = f"{target_date}T06:00:00+09:00"
    manifest["source"] = (
        f"Automated source-grounded cumulative pack through {target_date}; "
        f"current cumulative={manifest['cumulativeQuestionCount']}"
    )
    return manifest


def validate(pack: dict[str, Any], manifest: dict[str, Any], target_date: str) -> dict[str, Any]:
    qs = pack.get("questions", [])
    checks = []

    def ck(name, ok, detail=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    ck("total_2700", len(qs) == 2700, len(qs))
    tc = Counter(q.get("topic") for q in qs)
    dc = Counter(q.get("difficulty") for q in qs)
    cross = Counter((q.get("topic"), q.get("difficulty")) for q in qs)
    ck("36_topics_x75", all(tc.get(t) == 75 for t, _ in TOPICS), dict(tc))
    ck("5_difficulties_x540", all(dc.get(d) == 540 for d in DIFFICULTIES), dict(dc))
    ck("180_cells_x15", all(cross.get((t, d)) == 15 for t, _ in TOPICS for d in DIFFICULTIES))
    ck("unique_ids", len({q.get("id") for q in qs}) == 2700)
    ck("unique_questions", len({norm(q.get("question")) for q in qs}) == 2700)
    ck("unique_source_combos", len({tuple(sorted(map(str, q.get("sourceQuestionIds", [])))) for q in qs}) == 2700)
    ck("four_choices", all(isinstance(q.get("choices"), list) and len(q["choices"]) == 4 for q in qs))
    ck("answer_in_choices", all(q.get("answer") in q.get("choices", []) for q in qs))
    ck("accepted_answer", all(q.get("accepted") == [q.get("answer")] for q in qs))
    ck("source_count_matches_level", all(len(q.get("sourceQuestionIds", [])) == FACT_COUNT.get(q.get("difficulty")) for q in qs))
    ck("pack_date", pack.get("date") == target_date, pack.get("date"))
    ck("pack_counts", pack.get("questionCount") == 2700 and pack.get("cumulativeQuestionCount") == pack.get("previousCumulativeQuestionCount") + 2700)
    compact = target_date.replace("-", "")
    url = f"daily_knowledge_combo_{compact}_2700.json"
    ck("manifest_entry", any(isinstance(p, dict) and p.get("url") == url and p.get("questionCount") == 2700 for p in manifest.get("packs", [])))
    ck("manifest_date", manifest.get("updatedAt") == f"{target_date}T06:00:00+09:00", manifest.get("updatedAt"))
    ck("manifest_cumulative", int(manifest.get("cumulativeQuestionCount", -1)) == int(pack.get("cumulativeQuestionCount", -2)), manifest.get("cumulativeQuestionCount"))

    passed = sum(x["pass"] for x in checks)
    return {
        "schemaVersion": 1,
        "date": target_date,
        "generator": "knowledge-combo-source-grounded-auto-v1",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    try:
        dt.date.fromisoformat(args.date)
    except ValueError:
        print("Invalid --date; expected YYYY-MM-DD", file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    target_date = args.date
    compact = target_date.replace("-", "")
    pack_path = root / f"daily_knowledge_combo_{compact}_2700.json"
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)

    if pack_path.exists():
        pack = load_json(pack_path)
        print(f"{target_date}: dated 2700 pack already exists; preserving existing content")
    else:
        pack = generate(root, target_date, manifest)
        dump_json(pack_path, pack)

    manifest = update_manifest(manifest, pack, target_date)
    dump_json(manifest_path, manifest)
    report = validate(pack, manifest, target_date)
    dump_json(root / "AUTO_VALIDATION_REPORT.json", report)
    if report["status"] != "PASS":
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(f"PASS {report['passed']}/{report['total']} | 36x5x15=2700 | cumulative={manifest['cumulativeQuestionCount']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
