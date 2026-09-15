"""P5. Обогащение канонических документов, Qwen2.5-7B (FP8) через vLLM.

Вход: data/stories/canonical.csv (article_id, story_id, source, dt_utc, title, text).
Выход в data/enriched/: shard_XXXXX.parquet по 5000 строк, parts/*.jsonl для незаконченного шарда,
done_ids.txt, progress.json, run_meta.json. Сводка в data/reports/p5_summary.json и p5_distributions.csv.

Ответы пишутся в parts/shard_XXXXX.jsonl сразу, parquet создаётся когда шард собран полностью.
При перезапуске готовые шарды пропускаются, незаконченный дочитывается.

vllm serve RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic --served-model-name Qwen2.5-7B-Instruct \
--max-model-len 8192 --gpu-memory-utilization 0.65 --host 127.0.0.1 --port 8002

uv run python enrich.py --limit 1000  # проба
uv run python enrich.py
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from enrich_schema import (
    DATA,
    VLLM,
    FatalError,
    add_model_args,
    iter_rows,
    load_docs,
    load_prompt,
    run_meta,
    write_json,
)

PROGRESS_EVERY = 100  # обновлять progress.json каждые N документов

ENTITY_STRUCT = pa.struct([
    ("name", pa.string()),
    ("type", pa.string()),
    ("role", pa.string()),
    ("sentiment", pa.string()),
    ("normalized_id", pa.string()),
])

# схема явно, иначе на строках с ok=False (пустые поля) pyarrow выводит разные типы в разных шардах
SHARD_SCHEMA = pa.schema([
    ("article_id", pa.string()),
    ("story_id", pa.string()),
    ("source", pa.string()),
    ("dt_utc", pa.string()),
    ("summary", pa.string()),
    ("entities", pa.list_(ENTITY_STRUCT)),
    ("topic", pa.string()),
    ("event_type", pa.string()),
    ("is_opinion", pa.bool_()),
    ("confidence", pa.float32()),
    ("raw_response", pa.string()),
    ("ok", pa.bool_()),
    ("error", pa.string()),
    ("completion_tokens", pa.int32()),
    ("latency_s", pa.float32()),
])

FIELDS = [field.name for field in SHARD_SCHEMA]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path):
    """Читает jsonl незаконченного шарда, битая последняя строка отбрасывается."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # обрыв при записи
    return rows


def write_shard(path, rows):
    """Пишет шард в parquet (через .tmp), строки отсортированы по article_id."""
    rows = sorted(rows, key=lambda row: row["article_id"])
    table = pa.Table.from_pylist([{f: row.get(f) for f in FIELDS} for row in rows], SHARD_SCHEMA)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def collect_done(out):
    """Обработанные article_id по шардам и parts/*.jsonl (done_ids.txt может отставать)."""
    done = set()
    for path in sorted(out.glob("shard_*.parquet")):
        done.update(pq.read_table(path, columns=["article_id"])["article_id"].to_pylist())
    for path in sorted((out / "parts").glob("shard_*.jsonl")):
        done.update(row["article_id"] for row in read_jsonl(path))
    return done


def write_progress(path, state):
    """Прогресс в json, смотреть: watch -n 10 cat data/enriched/progress.json"""
    elapsed = time.perf_counter() - state["perf0"]
    done_now = state["done"] - state["done_at_start"]
    rate = done_now / elapsed if elapsed > 0 and done_now else 0.0
    left = state["total"] - state["done"]
    write_json(path, {
        "total": state["total"],
        "done": state["done"],
        "left": left,
        "percent": round(100 * state["done"] / state["total"], 2) if state["total"] else 0.0,
        "failed": state["failed"],
        "shards_done": state["shards_done"],
        "shards_total": state["shards_total"],
        "done_this_run": done_now,
        "docs_per_sec": round(rate, 2),
        "eta_minutes": round(left / rate / 60, 1) if rate else None,
        "elapsed_minutes": round(elapsed / 60, 1),
        "started_at": state["started_at"],
        "updated_at": now(),
        "source": state["source"],
        "out": state["out"],
    })


# колонки для сводки, без raw_response
SUMMARY_COLUMNS = ["ok", "topic", "event_type", "is_opinion", "confidence",
                   "entities", "completion_tokens"]


def load_rows(out):
    """Строки из шардов и parts для сводки."""
    rows = []
    for path in sorted(out.glob("shard_*.parquet")):
        rows.extend(pq.read_table(path, columns=SUMMARY_COLUMNS).to_pylist())
    for path in sorted((out / "parts").glob("shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def distribution(name, values):
    counter = Counter(values)
    total = sum(counter.values())
    return [
        {"field": name, "value": str(value), "count": count, "share": round(count / total, 4)}
        for value, count in counter.most_common()
    ]


def summary(out, reports, meta, gpu_price):
    """Сводка прогона: доля валидного JSON, распределения, скорость, стоимость."""
    rows = load_rows(out)
    if not rows:
        return None
    ok = [row for row in rows if row["ok"]]
    n, n_ok = len(rows), len(ok)

    entities = [entity for row in ok for entity in (row["entities"] or [])]
    per_doc = [len(row["entities"] or []) for row in ok]
    confidence = pd.Series([row["confidence"] for row in ok], dtype="float64")

    table = []
    table += distribution("topic", [row["topic"] for row in ok])
    table += distribution("event_type", [row["event_type"] for row in ok])
    table += distribution("is_opinion", [row["is_opinion"] for row in ok])
    table += distribution("entity_type", [entity["type"] for entity in entities])
    table += distribution("entity_role", [entity["role"] for entity in entities])
    table += distribution("entity_sentiment", [entity["sentiment"] for entity in entities])
    if n_ok:
        bins = [round(0.1 * i, 1) for i in range(11)]
        hist = pd.cut(confidence, bins=bins, include_lowest=True).value_counts().sort_index()
        table += [
            {"field": "confidence", "value": str(interval), "count": int(count),
             "share": round(count / n_ok, 4)}
            for interval, count in hist.items()
        ]
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(reports / "p5_distributions.csv", index=False)

    elapsed_h = meta.get("elapsed_minutes", 0) / 60
    result = {
        **meta,
        "documents": n,
        "valid_json_share": round(n_ok / n, 4),
        "failed": n - n_ok,
        "truncated_share": round(
            sum(row["completion_tokens"] >= meta["max_tokens"] for row in rows) / n, 4),
        "empty_entities_share": round(sum(k == 0 for k in per_doc) / n_ok, 4) if n_ok else None,
        "entities_per_doc": round(sum(per_doc) / n_ok, 2) if n_ok else None,
        "confidence_mean": round(float(confidence.mean()), 3) if n_ok else None,
        "gpu_hours": round(elapsed_h, 3),
        "cost_per_document": round(gpu_price * elapsed_h / n, 6) if gpu_price and n else None,
    }
    write_json(reports / "p5_summary.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(
        description="P5. Обогащение канонических документов, линия 2 (Qwen2.5-7B FP8)")
    ap.add_argument("--src", type=Path, default=DATA / "stories" / "canonical.csv")
    ap.add_argument("--out", type=Path, default=DATA / "enriched")
    ap.add_argument("--reports", type=Path, default=DATA / "reports")
    ap.add_argument("--shard", type=int, default=5000, help="документов в одном parquet")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N документов (проба)")
    ap.add_argument("--gpu-price", type=float, default=0.0, help="цена аренды GPU в час")
    ap.add_argument("--force", action="store_true",
                    help="продолжить прогон, даже если схема или промпт изменились")
    add_model_args(ap)
    args = ap.parse_args()

    parts_dir = args.out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "progress.json"
    meta_path = args.out / "run_meta.json"

    prompt = load_prompt(args.prompt)
    all_docs, docs = load_docs(args.src, args.limit)
    total = len(docs)
    if not total:
        raise SystemExit(f"{args.src} пуст")

    # не смешиваем шарды с разными версиями схемы/промпта
    meta = run_meta(args, prompt, {"shard": args.shard, "src": str(args.src)})
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = [key for key in ("schema_version", "prompt_version", "shard", "text_limit")
                   if old.get(key) != meta[key]]
        if changed and not args.force:
            raise SystemExit(
                f"в {args.out} уже есть прогон с другими параметрами: {', '.join(changed)}.\n"
                f"было {[old.get(k) for k in changed]}, стало {[meta[k] for k in changed]}.\n"
                f"очистите папку или запустите с --force")

    done_ids = collect_done(args.out)
    marker_path = args.out / "done_ids.txt"
    marker_path.write_text("".join(f"{aid}\n" for aid in sorted(done_ids)), encoding="utf-8")

    todo_total = sum(doc["article_id"] not in done_ids for doc in docs)
    print(f"документов: {total} из {len(all_docs)} в {args.src.name}")
    print(f"обработано ранее: {total - todo_total}, осталось: {todo_total}")
    print(f"схема {meta['schema_version']}, промпт {meta['prompt_version']}, "
          f"шард {args.shard}, запросов в полёте {args.concurrency}")

    state = {
        "total": total,
        "done": total - todo_total,
        "done_at_start": total - todo_total,
        "failed": 0,
        "shards_done": 0,
        "shards_total": (total + args.shard - 1) // args.shard,
        "started_at": now(),
        "perf0": time.perf_counter(),
        "source": str(args.src),
        "out": str(args.out),
    }

    if todo_total:
        client = VLLM(args.api, args.model, args.timeout, args.max_tokens)
        probe_doc = next(doc for doc in docs if doc["article_id"] not in done_ids)
        try:
            tokens = client.probe(prompt, probe_doc, args.text_limit)
        except FatalError as exc:
            raise SystemExit(
                f"{exc}\n\nЕсли сервер не принял схему (maxItems / minimum), перезапустите\n"
                f"vllm serve с бэкендом guidance, см. docs/run_enrichment.md")
        print(f"проба схемы прошла: {tokens} токенов в ответе\n")

        meta["started_at"] = state["started_at"]
        write_json(meta_path, meta)
        write_progress(progress_path, state)

        bar = tqdm(total=total, initial=state["done"], unit="док", smoothing=0.05)
        marker = marker_path.open("a", encoding="utf-8")
        try:
            for start in range(0, total, args.shard):
                shard_id = start // args.shard
                chunk = docs[start:start + args.shard]
                shard_path = args.out / f"shard_{shard_id:05d}.parquet"
                if shard_path.exists():
                    state["shards_done"] += 1
                    continue

                part_path = parts_dir / f"shard_{shard_id:05d}.jsonl"
                rows = read_jsonl(part_path)
                have = {row["article_id"] for row in rows}
                todo = [doc for doc in chunk if doc["article_id"] not in have]

                if todo:
                    bar.set_description(f"шард {shard_id:05d}")
                    with part_path.open("a", encoding="utf-8") as part:
                        for row in iter_rows(client, prompt, todo, args.concurrency, args.text_limit):
                            part.write(json.dumps(row, ensure_ascii=False) + "\n")
                            part.flush()
                            marker.write(row["article_id"] + "\n")
                            rows.append(row)
                            state["done"] += 1
                            state["failed"] += not row["ok"]
                            bar.update(1)
                            if state["done"] % PROGRESS_EVERY == 0:
                                marker.flush()
                                write_progress(progress_path, state)
                                bar.set_postfix(сбоев=state["failed"])

                # шард закрывается только полным; последний шард корпуса может быть короче,
                # но не при --limit
                is_tail = start + len(chunk) == len(all_docs)
                if len(rows) == len(chunk) and (len(chunk) == args.shard or is_tail):
                    write_shard(shard_path, rows)
                    part_path.unlink()
                    state["shards_done"] += 1
                marker.flush()
                write_progress(progress_path, state)
        except KeyboardInterrupt:
            print("\nпрервано, результат сохранён, при повторном запуске продолжит")
        finally:
            marker.close()
            bar.close()
            write_progress(progress_path, state)

    else:
        state["shards_done"] = len(list(args.out.glob("shard_*.parquet")))
        write_progress(progress_path, state)

    elapsed_min = round((time.perf_counter() - state["perf0"]) / 60, 1)
    meta.update({
        "finished_at": now(),
        "elapsed_minutes": elapsed_min,
        "documents_this_run": state["done"] - state["done_at_start"],
        "docs_per_sec": round((state["done"] - state["done_at_start"]) / max(
            time.perf_counter() - state["perf0"], 1e-9), 2),
    })
    write_json(meta_path, meta)

    result = summary(args.out, args.reports, meta, args.gpu_price)
    if result:
        print(f"\nдокументов в результате: {result['documents']}, "
              f"валидный JSON: {result['valid_json_share']:.1%}, "
              f"сбоев: {result['failed']}")
        if result["entities_per_doc"] is not None:
            print(f"сущностей на документ: {result['entities_per_doc']}, "
                  f"пустых списков: {result['empty_entities_share']:.1%}")
        print(f"ответ упёрся в max_tokens: {result['truncated_share']:.1%}")
        print(f"скорость: {meta['docs_per_sec']} док/с, GPU-часов: {result['gpu_hours']}")
        if result["cost_per_document"]:
            print(f"стоимость документа: {result['cost_per_document']}")
        print(f"сводка: {args.reports / 'p5_summary.json'}, "
              f"распределения: {args.reports / 'p5_distributions.csv'}")
    print(f"счётчик прогона: {progress_path}")


if __name__ == "__main__":
    main()
