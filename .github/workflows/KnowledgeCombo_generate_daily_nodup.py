#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any

ORIGINAL = Path(__file__).parent / ".github" / "workflows" / "KnowledgeCombo_generate_daily.py"
spec = importlib.util.spec_from_file_location("knowledge_combo_original", ORIGINAL)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load original generator: {ORIGINAL}")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

def render_question(topic_name: str, difficulty: str, combo: tuple[dict[str, str], ...]) -> str:
    blocks = [f"({orig.MARKERS[i]}) {a['prompt']}" for i, a in enumerate(combo)]
    return (
        f"[{topic_name}·{orig.DIFF_LABEL[difficulty]}] 다음 {len(combo)}개 물음의 답을 "
        f"(가)부터 순서대로 바르게 짝지은 것은?\n\n" + "\n".join(blocks)
    )

def generate(root: Path, target_date: str, manifest: dict[str, Any]) -> dict[str, Any]:
    compact = target_date.replace("-", "")
    pack_id = orig.next_pack_id(manifest, target_date)
    pools, historical_signatures, historical_questions = orig.build_sources(root)
    selected_global: set[tuple[str, ...]] = set()
    questions: list[dict[str, Any]] = []

    for topic, topic_name in orig.TOPICS:
        for difficulty in orig.DIFFICULTIES:
            n_facts = orig.FACT_COUNT[difficulty]
            chosen: list[tuple[dict[str, str], ...]] = []
            chosen_questions: list[str] = []
            rejected_text_duplicates = 0

            for broaden in (False, True):
                pool = orig.candidate_pool(pools, topic, difficulty, broaden)
                need = 15 - len(chosen)
                if need <= 0:
                    break

                candidate_budget = max(300, need * 40)
                more = orig.select_unused_combos(
                    pool,
                    n_facts,
                    candidate_budget,
                    target_date,
                    f"{topic}:{difficulty}:{'broad' if broaden else 'exact'}:text-dedup",
                    historical_signatures,
                    selected_global,
                )

                for combo in more:
                    sig = tuple(sorted(a["source_id"] for a in combo))
                    if sig in selected_global:
                        continue

                    question = render_question(topic_name, difficulty, combo)
                    nq = orig.norm(question)
                    if nq in historical_questions:
                        rejected_text_duplicates += 1
                        continue

                    chosen.append(combo)
                    chosen_questions.append(question)
                    selected_global.add(sig)
                    historical_questions.add(nq)
                    if len(chosen) >= 15:
                        break

                if len(chosen) >= 15:
                    break

            if len(chosen) != 15:
                exact_n = len(orig.candidate_pool(pools, topic, difficulty, False))
                broad_n = len(orig.candidate_pool(pools, topic, difficulty, True))
                raise RuntimeError(
                    f"{topic}/{difficulty}: only {len(chosen)}/15 text-unique new combos; "
                    f"historical-text rejects={rejected_text_duplicates}, "
                    f"atom pools exact={exact_n}, broad={broad_n}"
                )

            answer_pool = orig.candidate_pool(pools, topic, difficulty, True)
            for idx, (combo, question) in enumerate(zip(chosen, chosen_questions), 1):
                parts = [a["answer"] for a in combo]
                correct = " / ".join(parts)
                rng = random.Random(
                    int(orig.stable_hash(target_date, topic, difficulty, str(idx))[:16], 16)
                )
                choices = orig.distractors(parts, answer_pool, rng) + [correct]
                rng.shuffle(choices)
                explanation = "정답 대응은 " + ", ".join(
                    f"({orig.MARKERS[i]}) {a['answer']}" for i, a in enumerate(combo)
                ) + "입니다."
                questions.append({
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
                })

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

orig.generate = generate
raise SystemExit(orig.main())
