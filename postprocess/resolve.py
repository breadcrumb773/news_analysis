"""Шаги 4-6: батчи, запросы к LLM, проверка ответов."""

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from enrich_schema import ENTITY_TYPES, ROOT
from postprocess.common import BATCH_SIZE, log

PROMPT_PATH = ROOT / "configs" / "prompts" / "entity_merge.txt"
RETRIES = 8  # пауза удваивается, в сумме > 4 мин, чтобы пережить rate limit по TPM
TIMEOUT = 120  # сек

# схема structured output
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "confidence": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["groups", "confidence"],
    "additionalProperties": False,
}


def make_batches(groups, norm):
    """Шаг 4. Группы кандидатов -> батчи примерно по BATCH_SIZE форм.

    Один тип на батч. Группа целиком попадает в один батч. group_of хранит номер
    группы для каждой формы, используется при проверке ответа.
    """
    batches = []
    for type_ in ENTITY_TYPES:
        forms, group_of = [], []
        for gid, group in enumerate(groups):
            if norm.at[group[0], "type"] != type_:
                continue
            if forms and len(forms) + len(group) > BATCH_SIZE:
                batches.append({"type": type_, "forms": forms, "group_of": group_of})
                forms, group_of = [], []
            forms.extend(group)
            group_of.extend([gid] * len(group))
        if forms:
            batches.append({"type": type_, "forms": forms, "group_of": group_of})
    for k, batch in enumerate(batches):
        batch["batch_id"] = f"b_{k:05d}"
    return batches


def build_prompt(template, batch, norm):
    """Промпт для батча, строка на форму: индекс, alias_key, n_stories, заголовок."""
    lines = []
    for i, fid in enumerate(batch["forms"], 1):
        title = norm.at[fid, "sample_title"]
        title = title if isinstance(title, str) else ""
        lines.append(f"{i}. {norm.at[fid, 'alias_key']} ({norm.at[fid, 'n_stories']}) | {title}")
    return template.replace("{type}", batch["type"]).replace("{forms}", "\n".join(lines))


def ask_model(prompt, api, model, key):
    """Запрос к OpenAI-совместимому /v1/chat/completions, возвращает (content, usage).

    429 и 5xx - повтор с экспоненциальной паузой, остальные ошибки сразу исключение.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "groups", "schema": ANSWER_SCHEMA, "strict": True},
        },
    }
    headers = {"Authorization": f"Bearer {key}"}
    error = ""
    for attempt in range(RETRIES):
        try:
            resp = requests.post(api, json=payload, headers=headers, timeout=TIMEOUT)
        except requests.RequestException as exc:
            error = str(exc)
        else:
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"], data.get("usage", {})
            error = f"{resp.status_code}: {resp.text[:300]}"
            if resp.status_code != 429 and resp.status_code < 500:
                raise RuntimeError(error)
        # jitter, чтобы потоки не повторяли одновременно
        time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"не удалось за {RETRIES} попыток: {error}")


def load_cache(path):
    """Кэш ответов, ключ - хэш модели и промпта."""
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # недописанная строка
            cache[row["key"]] = row
    return cache


def ask_all(batches, prompts, paths, args, api_key):
    """Шаг 5. Отправка батчей, которых нет в кэше, в args.concurrency потоков.

    Ответы сразу пишутся в data/entities/llm_responses.jsonl.
    """
    cache_path = paths.out / "llm_responses.jsonl"
    cache = load_cache(cache_path)
    key_of = {b["batch_id"]: hashlib.sha256(f"{args.llm_model}\n{prompts[b['batch_id']]}".encode())
              .hexdigest()[:16] for b in batches}

    answers, usage = {}, Counter()
    for b in batches:
        row = cache.get(key_of[b["batch_id"]])
        if row:
            answers[b["batch_id"]] = row["raw"]
            usage.update({k: v for k, v in (row.get("usage") or {}).items() if isinstance(v, int)})

    todo = [b for b in batches if b["batch_id"] not in answers]
    if args.max_batches:
        todo = todo[:args.max_batches]
    log.info(f"шаг 5: батчей {len(batches)}, из кэша {len(answers)}, к отправке {len(todo)}")
    failed = []
    if not todo:
        return answers, failed, usage

    paths.out.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as cache_file:

        def save(batch, raw, used):
            row = {"key": key_of[batch["batch_id"]], "batch_id": batch["batch_id"],
                   "model": args.llm_model, "raw": raw, "usage": used}
            cache_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache_file.flush()
            answers[batch["batch_id"]] = raw
            usage.update({k: v for k, v in used.items() if isinstance(v, int)})

        # первый батч отдельно, чтобы ошибка ключа/модели падала сразу
        first = todo[0]
        try:
            save(first, *ask_model(prompts[first["batch_id"]], args.llm_api, args.llm_model, api_key))
        except RuntimeError as exc:
            raise SystemExit(f"ошибка запроса к LLM: {exc}")

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(ask_model, prompts[b["batch_id"]], args.llm_api,
                                   args.llm_model, api_key): b for b in todo[1:]}
            for future in tqdm(as_completed(futures), total=len(futures), desc="шаг 5", unit="батч"):
                batch = futures[future]
                try:
                    save(batch, *future.result())
                except Exception as exc:
                    log.info(f"  батч {batch['batch_id']} без ответа: {exc}")
                    failed.append(batch["batch_id"])
    return answers, failed, usage


def pick_owner(groups, counts, n):
    """Для индексов, попавших в несколько групп: номер группы, где индекс останется.

    Выбирается самая маленькая группа, при равенстве первая. На пробном прогоне дубли
    были в основном между маленькими группами и одной большой сборной.
    """
    owner, width = {}, {}
    for number, group in enumerate(groups):
        size = len({i for i in group if isinstance(i, int) and 1 <= i <= n})
        for i in group:
            if isinstance(i, int) and counts[i] > 1 and size < width.get(i, n + 1):
                owner[i], width[i] = number, size
    return owner


def check_answer(raw, batch, norm):
    """Шаг 6. Разбор ответа LLM: группы форм батча и журнал вето.

    Причины вето:
    invented - индекса нет в батче
    duplicate - индекс в нескольких группах, остаётся в одной (pick_owner)
    lost - индекс не попал ни в одну группу
    cross_group - в группе формы из разных групп кандидатов, группа делится обратно по ним
    """
    forms, group_of = batch["forms"], batch["group_of"]
    n = len(forms)
    veto = []

    def note(i, reason):
        fid = forms[i - 1] if isinstance(i, int) and 1 <= i <= n else None
        veto.append({"batch_id": batch["batch_id"], "index": i, "form_id": fid,
                     "alias_key": norm.at[fid, "alias_key"] if fid is not None else None,
                     "reason": reason})

    try:
        answer = json.loads(raw)
        groups = [list(g) for g in answer["groups"]]
        confidence = list(answer.get("confidence") or [])
    except (json.JSONDecodeError, KeyError, TypeError):
        # невалидный JSON: все формы батча остаются одиночными
        note(None, "invalid_json")
        return [{"batch_id": batch["batch_id"], "form_ids": [fid], "confidence": None}
                for fid in forms], veto

    counts = Counter(i for g in groups for i in g if isinstance(i, int))
    owner = pick_owner(groups, counts, n)
    accepted = []
    for number, group in enumerate(groups):
        conf = confidence[number] if number < len(confidence) else None
        conf = float(conf) if isinstance(conf, (int, float)) else None

        members, seen = [], set()
        for i in group:
            if not isinstance(i, int) or not 1 <= i <= n:
                note(i, "invented")
            elif i in seen:
                continue  # повтор внутри группы
            elif counts[i] == 1 or owner[i] == number:
                if counts[i] > 1:
                    note(i, "duplicate")
                members.append(i)
                seen.add(i)
            # в других группах дубль пропускаем

        by_group = defaultdict(list)
        for i in members:
            by_group[group_of[i - 1]].append(forms[i - 1])
        if len(by_group) > 1:
            for i in members:
                note(i, "cross_group")
        for part in by_group.values():
            accepted.append({"batch_id": batch["batch_id"], "form_ids": part, "confidence": conf})

    placed = {fid for g in accepted for fid in g["form_ids"]}
    for i, fid in enumerate(forms, 1):
        if fid not in placed:
            note(i, "duplicate" if counts[i] > 1 else "lost")
            accepted.append({"batch_id": batch["batch_id"], "form_ids": [fid], "confidence": None})
    return accepted, veto


def resolve(groups, norm, paths, args, api_key):
    """Шаги 4-6. Возвращает принятые группы, вето и статистику."""
    batches = make_batches(groups, norm)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    prompts = {b["batch_id"]: build_prompt(template, b, norm) for b in batches}
    log.info(f"шаг 4: групп {len(groups)} -> батчей {len(batches)}")

    if args.no_llm:
        log.info("шаг 5 пропущен (--no-llm): все группы остаются несобранными")
        answers, failed, usage = {}, [], Counter()
    else:
        answers, failed, usage = ask_all(batches, prompts, paths, args, api_key)

    accepted, veto = [], []
    for batch in batches:
        raw = answers.get(batch["batch_id"])
        if raw is None:
            # нет ответа: формы одиночные, batch_id сохраняем
            accepted.extend({"batch_id": batch["batch_id"], "form_ids": [fid], "confidence": None}
                            for fid in batch["forms"])
            continue
        ok, bad = check_answer(raw, batch, norm)
        accepted.extend(ok)
        veto.extend(bad)

    veto = pd.DataFrame(veto, columns=["batch_id", "index", "form_id", "alias_key", "reason"])
    veto["form_id"] = veto["form_id"].astype("Int64")  # для invented form_id = NA
    stats = {
        "batches": len(batches),
        "batches_answered": len(answers),
        "batches_failed": len(failed),
        "batches_not_sent": len(batches) - len(answers) - len(failed),
        "merged_groups": sum(len(g["form_ids"]) > 1 for g in accepted),
        "veto": veto["reason"].value_counts().to_dict(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }
    log.info(f"шаг 6: ответов {len(answers)}, сбоев {len(failed)}, "
             f"групп на слияние {stats['merged_groups']}, вето {stats['veto']}")
    return accepted, veto, stats
