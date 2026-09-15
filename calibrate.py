"""P0. Калибровка схемы, промпта и параметров генерации перед P5.

Прогон 100 случайных статей через vLLM (temperature=0). Считается: доля валидного JSON,
распределения topic/event_type/тональности, скорость и оценка времени на корпус,
доля сущностей, которых нет во входе. Для ручной проверки выписываются примеры.
С --full дополнительно прогон на полном тексте и сравнение с усечением до 1500 символов.

После правки enrich_schema.py или configs/prompts/extraction.txt запускается заново,
каждый запуск добавляет строку в data/reports/p0_protocol.jsonl.

Вход: data/clean/news_2016.csv (нужны title, text).
Выход в data/reports/: p0_sample.jsonl, p0_manual.md, p0_distributions.csv, p0_protocol.jsonl, p0_last.json.
vLLM на порту 8002 (docs/run_enrichment.md).

uv run python calibrate.py --note "первый цикл"
uv run python calibrate.py --full --note "проверка усечения"
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from enrich_schema import (
    DATA,
    EVENT_TYPES,
    TOPICS,
    VLLM,
    FatalError,
    add_model_args,
    build_input,
    iter_rows,
    load_docs,
    load_prompt,
    run_meta,
    write_json,
)

FULL_TEXT = 1_000_000  # фактически без усечения

# пороги для замечаний, при превышении смотреть руками
TOP_SHARE_MAX = 0.60  # доля самого частого topic / event_type
NEUTRAL_MAX = 0.80  # доля neu у сущностей
EMPTY_ENTITIES_MAX = 0.10  # доля документов без сущностей
UNSEEN_ENTITY_MAX = 0.05  # доля сущностей, не найденных во входе
TRUNCATED_MAX = 0.01  # доля ответов с обрывом по max_tokens


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(text):
    return text.lower().replace("ё", "е")


def seen_in_text(name, haystack):
    """Грубая проверка, что сущность есть во входе.

    Сначала прямое вхождение, иначе по основам (первые 5 букв слов длиной от 4),
    чтобы учесть смену падежа.
    """
    name = normalize(name).strip()
    if not name:
        return False
    if name in haystack:
        return True
    stems = [word[:5] for word in re.findall(r"\w+", name) if len(word) >= 4]
    return bool(stems) and all(stem in haystack for stem in stems)


def run_pass(client, prompt, docs, concurrency, text_limit, desc):
    """Прогон выборки, возвращает ({article_id: row}, секунды)."""
    started = time.perf_counter()
    rows = {}
    for row in tqdm(
        iter_rows(client, prompt, docs, concurrency, text_limit),
        total=len(docs), desc=desc, unit="док",
    ):
        rows[row["article_id"]] = row
    return rows, time.perf_counter() - started


def distribution(name, values):
    counter = Counter(values)
    total = sum(counter.values()) or 1
    return [
        {"field": name, "value": str(value), "count": count, "share": round(count / total, 4)}
        for value, count in counter.most_common()
    ]


def top_share(values):
    counter = Counter(values)
    total = sum(counter.values())
    if not total:
        return None, 0.0
    value, count = counter.most_common(1)[0]
    return value, count / total


def truncation_diff(short_rows, full_rows):
    """Сравнение прогонов с усечением и на полном тексте: сущности, topic, event_type, summary."""
    ids = [aid for aid in short_rows if aid in full_rows
           and short_rows[aid]["ok"] and full_rows[aid]["ok"]]
    if not ids:
        return None

    topic_changed = event_changed = opinion_changed = any_changed = 0
    jaccards, len_short, len_full = [], [], []
    for aid in ids:
        a, b = short_rows[aid], full_rows[aid]
        names_a = {normalize(e["name"]) for e in a["entities"]}
        names_b = {normalize(e["name"]) for e in b["entities"]}
        union = names_a | names_b
        jaccard = len(names_a & names_b) / len(union) if union else 1.0
        jaccards.append(jaccard)
        len_short.append(len(a["summary"] or ""))
        len_full.append(len(b["summary"] or ""))
        topic_changed += a["topic"] != b["topic"]
        event_changed += a["event_type"] != b["event_type"]
        opinion_changed += a["is_opinion"] != b["is_opinion"]
        any_changed += a["topic"] != b["topic"] or a["event_type"] != b["event_type"] or jaccard < 1
    n = len(ids)
    return {
        "documents": n,
        "topic_changed": round(topic_changed / n, 4),
        "event_type_changed": round(event_changed / n, 4),
        "is_opinion_changed": round(opinion_changed / n, 4),
        "entities_jaccard_mean": round(sum(jaccards) / n, 4),
        "entities_changed": round(sum(j < 1 for j in jaccards) / n, 4),
        "summary_len_short": round(sum(len_short) / n, 1),
        "summary_len_full": round(sum(len_full) / n, 1),
        "any_changed": round(any_changed / n, 4),
    }


def write_manual(path, rows, docs_by_id, text_limit, count):
    """Markdown с примерами для ручной проверки."""
    lines = [
        "# P0. Выборка для ручной проверки",
        "",
        "Смотрится: правильность полей, отсутствие галлюцинаций и выдуманных сущностей.",
        "",
    ]
    for aid in sorted(rows)[:count]:
        row, doc = rows[aid], docs_by_id[aid]
        lines += [f"## {aid} — {doc['title']}", ""]
        if not row["ok"]:
            lines += [f"ответ не разобран: {row['error']}", "", f"```\n{row['raw_response'][:1500]}\n```", ""]
            continue
        entities = "\n".join(
            f"- {e['name']} — {e['type']}, {e['role']}, {e['sentiment']} -> {e['normalized_id']}"
            for e in row["entities"]
        ) or "- (пусто)"
        lines += [
            "**вход**",
            "",
            "```",
            build_input(doc["title"], doc["text"], text_limit),
            "```",
            "",
            f"**summary**: {row['summary']}",
            "",
            f"**topic**: {row['topic']}  ·  **event_type**: {row['event_type']}  ·  "
            f"**is_opinion**: {row['is_opinion']}  ·  **confidence**: {row['confidence']}",
            "",
            "**сущности**",
            "",
            entities,
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="P0. Калибровка обогащения")
    ap.add_argument("--src", type=Path, default=DATA / "clean" / "news_2016.csv")
    ap.add_argument("--reports", type=Path, default=DATA / "reports")
    ap.add_argument("--n", type=int, default=100, help="сколько статей взять")
    ap.add_argument("--seed", type=int, default=0, help="фиксирует выборку между циклами")
    ap.add_argument("--full", action="store_true",
                    help="второй прогон на полном тексте: проверка достаточности 1500 знаков")
    ap.add_argument("--manual", type=int, default=15, help="сколько статей выписать для ручной проверки")
    ap.add_argument("--corpus", type=int, default=0,
                    help="размер корпуса для экстраполяции скорости (по умолчанию весь файл)")
    ap.add_argument("--note", default="", help="что менялось в этом цикле, пойдёт в протокол")
    add_model_args(ap)
    args = ap.parse_args()

    prompt = load_prompt(args.prompt)
    all_docs, _ = load_docs(args.src)
    if len(all_docs) < args.n:
        raise SystemExit(f"в {args.src} всего {len(all_docs)} строк, а нужно {args.n}")

    # фиксированный seed, чтобы циклы сравнивались на одной выборке
    docs = sorted(random.Random(args.seed).sample(all_docs, args.n),
                  key=lambda doc: doc["article_id"])
    docs_by_id = {doc["article_id"]: doc for doc in docs}
    corpus = args.corpus or len(all_docs)

    meta = run_meta(args, prompt, {"n": args.n, "seed": args.seed, "src": str(args.src)})
    print(f"схема {meta['schema_version']}, промпт {meta['prompt_version']}, "
          f"модель {args.model}")
    print(f"статей в выборке: {len(docs)} из {len(all_docs)}\n")

    client = VLLM(args.api, args.model, args.timeout, args.max_tokens)
    try:
        client.probe(prompt, docs[0], args.text_limit)
    except FatalError as exc:
        raise SystemExit(
            f"{exc}\n\nЕсли сервер не принял схему (maxItems / minimum), перезапустите\n"
            f"vllm serve с бэкендом guidance, см. docs/run_enrichment.md")

    rows, elapsed = run_pass(client, prompt, docs, args.concurrency, args.text_limit, "1500 знаков")
    full_rows = {}
    if args.full:
        full_rows, _ = run_pass(client, prompt, docs, args.concurrency, FULL_TEXT, "полный текст")

    ok = [row for row in rows.values() if row["ok"]]
    n, n_ok = len(rows), len(ok)
    valid_share = n_ok / n if n else 0.0
    rate = n / elapsed if elapsed else 0.0
    flags = []

    # распределения
    entities = [entity for row in ok for entity in row["entities"]]
    per_doc = [len(row["entities"]) for row in ok]
    empty_share = sum(k == 0 for k in per_doc) / n_ok if n_ok else 1.0
    topic_top, topic_share = top_share([row["topic"] for row in ok])
    event_top, event_share = top_share([row["event_type"] for row in ok])
    sent_top, sent_share = top_share([e["sentiment"] for e in entities])
    opinion_share = sum(row["is_opinion"] for row in ok) / n_ok if n_ok else 0.0
    truncated_share = sum(
        row["completion_tokens"] >= args.max_tokens for row in rows.values()) / n if n else 0.0

    # сущности, которых нет во входе
    unseen = 0
    for row in ok:
        haystack = normalize(build_input(
            docs_by_id[row["article_id"]]["title"],
            docs_by_id[row["article_id"]]["text"],
            args.text_limit,
        ))
        unseen += sum(not seen_in_text(entity["name"], haystack) for entity in row["entities"])
    unseen_share = unseen / len(entities) if entities else 0.0

    if valid_share < 1.0:
        flags.append(f"валидный JSON {valid_share:.1%}, а должно быть 100%")
    if topic_share > TOP_SHARE_MAX:
        flags.append(f"topic вырожден: «{topic_top}» на {topic_share:.0%} выборки")
    if event_share > TOP_SHARE_MAX:
        flags.append(f"event_type вырожден: «{event_top}» на {event_share:.0%} выборки")
    if sent_top == "neu" and sent_share > NEUTRAL_MAX:
        flags.append(f"тональность вырождена: neu на {sent_share:.0%} сущностей, "
                     f"проверить критерии pos/neg в промпте")
    if empty_share > EMPTY_ENTITIES_MAX:
        flags.append(f"пустой список сущностей у {empty_share:.0%} документов")
    if unseen_share > UNSEEN_ENTITY_MAX:
        flags.append(f"{unseen_share:.0%} сущностей не найдено во входном фрагменте, "
                     f"возможны выдуманные")
    if truncated_share > TRUNCATED_MAX:
        flags.append(f"{truncated_share:.0%} ответов упёрлись в max_tokens={args.max_tokens}")
    if len(set(row["topic"] for row in ok)) < len(TOPICS) / 3:
        flags.append("больше двух третей списка topic не использовано ни разу")

    diff = truncation_diff(rows, full_rows) if args.full else None
    if diff and diff["any_changed"] > 0.25:
        flags.append(f"усечение до {args.text_limit} знаков меняет результат "
                     f"у {diff['any_changed']:.0%} статей")

    print(f"\nвалидный JSON:        {n_ok}/{n} ({valid_share:.1%})")
    print(f"скорость:             {rate:.2f} док/с при {args.concurrency} запросах в полёте")
    print(f"на {corpus} документов: {corpus / rate / 3600:.1f} ч" if rate else "")
    print(f"сущностей на документ: {sum(per_doc) / n_ok:.1f}" if n_ok else "")
    print(f"пустых списков:       {empty_share:.1%}")
    print(f"нет во входе:         {unseen_share:.1%} сущностей")
    print(f"мнений (is_opinion):  {opinion_share:.1%}")
    print(f"упёрлись в max_tokens: {truncated_share:.1%}")

    print("\nраспределение topic")
    for item in distribution("topic", [row["topic"] for row in ok])[:12]:
        print(f"  {item['value']:<28} {item['count']:>4}  {item['share']:.1%}")
    print("распределение event_type")
    for item in distribution("event_type", [row["event_type"] for row in ok])[:15]:
        print(f"  {item['value']:<28} {item['count']:>4}  {item['share']:.1%}")
    print("тональность сущностей")
    for item in distribution("entity_sentiment", [e["sentiment"] for e in entities]):
        print(f"  {item['value']:<28} {item['count']:>4}  {item['share']:.1%}")

    if diff:
        print(f"\nусечение до {args.text_limit} знаков против полного текста, "
              f"{diff['documents']} статей")
        print(f"  topic меняется         {diff['topic_changed']:.1%}")
        print(f"  event_type меняется    {diff['event_type_changed']:.1%}")
        print(f"  состав сущностей       {diff['entities_changed']:.1%} "
              f"(средний Жаккар {diff['entities_jaccard_mean']:.2f})")
        print(f"  длина summary          {diff['summary_len_short']:.0f} -> "
              f"{diff['summary_len_full']:.0f} знаков")
        print(f"  меняется хоть что-то   {diff['any_changed']:.1%}")

    # сохранение
    args.reports.mkdir(parents=True, exist_ok=True)
    with (args.reports / "p0_sample.jsonl").open("w", encoding="utf-8") as f:
        for aid in sorted(rows):
            record = dict(rows[aid])
            if aid in full_rows:
                record["full_text_response"] = full_rows[aid]["raw_response"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    table = (distribution("topic", [row["topic"] for row in ok])
             + distribution("event_type", [row["event_type"] for row in ok])
             + distribution("is_opinion", [row["is_opinion"] for row in ok])
             + distribution("entity_type", [e["type"] for e in entities])
             + distribution("entity_role", [e["role"] for e in entities])
             + distribution("entity_sentiment", [e["sentiment"] for e in entities]))
    pd.DataFrame(table).to_csv(args.reports / "p0_distributions.csv", index=False)
    write_manual(args.reports / "p0_manual.md", rows, docs_by_id, args.text_limit, args.manual)

    record = {
        "at": now(),
        "note": args.note,
        **meta,
        "topics": len(TOPICS),
        "event_types": len(EVENT_TYPES),
        "valid_json_share": round(valid_share, 4),
        "docs_per_sec": round(rate, 2),
        "corpus": corpus,
        "corpus_hours": round(corpus / rate / 3600, 2) if rate else None,
        "entities_per_doc": round(sum(per_doc) / n_ok, 2) if n_ok else None,
        "empty_entities_share": round(empty_share, 4),
        "unseen_entities_share": round(unseen_share, 4),
        "truncated_share": round(truncated_share, 4),
        "topic_top": [topic_top, round(topic_share, 4)],
        "event_type_top": [event_top, round(event_share, 4)],
        "sentiment_top": [sent_top, round(sent_share, 4)],
        "is_opinion_share": round(opinion_share, 4),
        "truncation_check": diff,
        "flags": flags,
    }
    with (args.reports / "p0_protocol.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(args.reports / "p0_last.json", record)

    print("\nЗАМЕЧАНИЯ")
    if flags:
        for flag in flags:
            print(f"  - {flag}")
        print("\nнужна правка enrich_schema.py или configs/prompts/extraction.txt и повторный запуск")
    else:
        print("  замечаний нет")
    print(f"\nответы: {args.reports / 'p0_sample.jsonl'}")
    print(f"ручная проверка: {args.reports / 'p0_manual.md'}")
    print(f"протокол циклов: {args.reports / 'p0_protocol.jsonl'}")


if __name__ == "__main__":
    main()
