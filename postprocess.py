"""Постобработка обогащения: поверхностные формы сущностей -> канонические сущности.

Код по шагам в postprocess/:
load.py - загрузка шардов P5 и текстов, шаг 1 (частотный словарь форм)
normalize.py - шаг 2, чистка и нормализация форм
candidates.py - шаг 3, кандидаты на слияние: канал A (эмбеддинги) и канал C (нормализация из P5)
resolve.py - шаги 4-6, батчи, запросы к LLM, проверка ответов
assemble.py - шаг 7, Union-Find, entity_id, итоговые таблицы
output.py - сохранение parquet, дополненные шарды, отчёты

Вход: data/enriched/shard_*.parquet (P5), data/stories/canonical.csv (P4).
Выход:
data/entities/ - mentions, forms, forms_norm, entities (parquet), кэш векторов form_vectors.npy
и кэш ответов LLM llm_responses.jsonl
data/enriched_linked/shard_*.parquet - шарды P5 с alias_key, entity_id, canonical, method
data/reports/postprocess_* - лог, summary.json, compression.csv, top20.csv, veto_log.csv

Нужно: vLLM с USER-bge-m3 на 127.0.0.1:8000 (docs/run_embeddings.md) для канала A
и ключ в OPENAI_API_KEY. Векторы и ответы LLM кэшируются.

Запуск:
uv run python postprocess.py
uv run python postprocess.py --max-batches 3  # проба на 3 батчах
uv run python postprocess.py --no-emb --no-llm  # без эмбеддера и LLM
"""

import argparse
import os
import time

from enrich_schema import DATA
from postprocess.assemble import assemble
from postprocess.candidates import find_candidates
from postprocess.common import Paths, setup_log
from postprocess.load import explode_mentions, load_articles, load_enriched, surface_forms
from postprocess.normalize import normalize_forms
from postprocess.output import link_enriched, save_tables, write_reports
from postprocess.resolve import resolve


def main():
    ap = argparse.ArgumentParser(description="Постобработка обогащения")
    ap.add_argument("--data", default=DATA, help="корень data/")
    ap.add_argument("--emb-api", default="http://127.0.0.1:8000/v1/embeddings")
    ap.add_argument("--emb-model", default="USER-bge-m3")
    ap.add_argument("--llm-api", default="https://api.openai.com/v1/chat/completions",
                    help="любой OpenAI-совместимый /v1/chat/completions")
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    ap.add_argument("--llm-key-env", default="OPENAI_API_KEY",
                    help="переменная окружения с ключом облачной модели")
    ap.add_argument("--concurrency", type=int, default=12, help="запросов к модели одновременно")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="отправить в модель не больше N новых батчей (проба)")
    ap.add_argument("--no-emb", action="store_true", help="без канала A")
    ap.add_argument("--no-llm", action="store_true", help="без облачной модели")
    args = ap.parse_args()

    paths = Paths(args.data)
    log = setup_log(paths.reports)
    started = time.perf_counter()
    log.info("=" * 70)
    log.info(f"постобработка, старт: data={paths.data}, модель {args.llm_model}, "
             f"канал A {'выкл' if args.no_emb else 'вкл'}, облако {'выкл' if args.no_llm else 'вкл'}")

    # проверка ключа до загрузки данных
    api_key = os.environ.get(args.llm_key_env, "")
    if not args.no_llm and not api_key:
        raise SystemExit(f"нет ключа в переменной {args.llm_key_env} (или запустите с --no-llm)")

    # загрузка, шаг 1
    enriched = load_enriched(paths.enriched)
    articles, inputs = load_articles(paths.canonical)
    mentions = explode_mentions(enriched, inputs)
    del enriched, inputs  # освобождаем память
    forms = surface_forms(mentions, articles)

    forms, norm, mentions = normalize_forms(forms, mentions, articles)

    groups, edges, candidate_stats = find_candidates(norm, paths, args)

    accepted, veto, llm_stats = resolve(groups, norm, paths, args, api_key)

    mentions, forms, norm, entities, assemble_stats = assemble(forms, norm, mentions,
                                                               accepted, veto, edges)

    # сохранение
    save_tables(paths, mentions, forms, norm, entities, veto)
    link_enriched(paths, forms, norm)
    stats = {
        "llm_model": None if args.no_llm else args.llm_model,
        "channel_a": not args.no_emb,
        "mentions": len(mentions),
        "grounded_share": round(float(mentions["grounded"].mean()), 4),
        **assemble_stats,
        "candidates": candidate_stats,
        "llm": llm_stats,
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    write_reports(paths, forms, norm, entities, stats)
    log.info(f"постобработка готова за {stats['elapsed_minutes']} мин")


if __name__ == "__main__":
    main()
