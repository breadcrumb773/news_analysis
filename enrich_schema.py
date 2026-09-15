"""Схема обогащения, промпт и клиент vLLM. Используется в calibrate.py (P0) и enrich.py (P5).

Длины ограничены (MAX_ENTITIES, max_length), в JSON-схеме это maxItems/maxLength.
Без ограничений ответ с длинным списком сущностей упирался в max_tokens и JSON обрывался.

SCHEMA_VERSION и prompt_version - хэши содержимого, пишутся в метаданные прогона.
"""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
PROMPT_PATH = ROOT / "configs" / "prompts" / "extraction.txt"

TEXT_LIMIT = 1500  # символов текста после заголовка, как в P2
MAX_TOKENS = 1024  # 20 сущностей помещаются
MAX_ENTITIES = 20
RETRIES = 5  # пауза между попытками удваивается

# Списки классов, подбирались в циклах P0.
# Значение - описание класса для промпта. В схему идут только ключи, на SCHEMA_VERSION описание не влияет.
TOPIC_HINTS = {
    "политика": "власть, выборы, партии, госуправление, международные отношения, санкции",
    "экономика и бизнес": "рынки, компании, финансы, бюджет, цены, валюта, налоги, отрасли; "
                          "не спорт и не оборонзаказ, даже если речь о деньгах",
    "общество": "социальная сфера, образование, ЖКХ, пенсии и льготы, религия и церковь, "
                "погода, экология, благотворительность и общественные акции, "
                "быт и нравы, городская жизнь, знаменитости, курьёзы; "
                "всё, что не подходит под остальные рубрики",
    "происшествия": "пожары, аварии, катастрофы, землетрясения и другие стихийные бедствия, "
                    "несчастные случаи, отравления",
    "армия и конфликты": "только военное: армия и флот, военная техника и оборонзаказ, учения, "
                         "боевые действия, войны, теракты. Смерть или отставка "
                         "чиновника, спортивный клуб, стихийное бедствие сюда не относятся",
    "право и криминал": "преступления, задержания, следствие, суды, приговоры, "
                        "коррупция, шпионаж, работа полиции и СК",
    "наука и технологии": "исследования, открытия, космос, IT, гаджеты, интернет",
    "медицина и здоровье": "болезни, эпидемии, лечение, здравоохранение, лекарства",
    "культура": "искусство, кино, музыка, литература, театр, памятники, наследие, шоу-бизнес",
    "спорт": "соревнования, матчи, клубы, спортсмены, трансферы, допинг",
    "транспорт и инфраструктура": "дороги, авиа- и железнодорожное сообщение, "
                                  "строительство, энергосети, связь",
}
# "прочее" убрано в цикле 3, модель слишком часто его выбирала. Вместо него "общество".
TOPICS = tuple(TOPIC_HINTS)

EVENT_TYPE_HINTS = {
    "заявление": "новость в самом высказывании: оценка, позиция, обещание, обвинение, "
                 "критика. Если кто-то сообщает о событии, выбирай тип события",
    "встреча и переговоры": "визиты, переговоры, саммиты, телефонные разговоры",
    "решение властей": "законы, указы, постановления, запреты, разрешения, регулирование",
    "назначение или отставка": "кадровые решения, уход с поста, смерть должностного лица",
    "сделка или инвестиция": "покупки, слияния, контракты, кредиты, вложения",
    "отчёт или статистика": "данные, показатели, рейтинги, результаты исследований и опросов",
    "происшествие или авария": "пожары, ДТП, катастрофы, стихийные бедствия, гибель людей",
    "преступление": "совершённое или предотвращённое преступление, задержание подозреваемых",
    "суд и расследование": "уголовные дела, обыски, допросы, суды, приговоры",
    "протест или акция": "митинги, забастовки, пикеты, общественные и благотворительные акции",
    "боевые действия": "удары, бои, обстрелы, военные операции, теракты",
    "спортивное событие": "матчи, турниры, результаты, рекорды",
    "культурное событие": "премьеры, выставки, фестивали, премии, концерты",
    "анонс или план": "то, что только запланировано или ожидается",
    "прочее": "только если не подходит ни один тип выше",
}
EVENT_TYPES = tuple(EVENT_TYPE_HINTS)

ENTITY_TYPES = ("ORG", "PER", "LOC", "GPE", "PRODUCT", "EVENT")
ROLES = ("actor", "object", "mentioned")
SENTIMENTS = ("pos", "neu", "neg", "unknown")


class Entity(BaseModel):
    """Одно упоминание сущности: поверхностная форма и то, что модель о ней думает."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=120, description="форма ровно как в тексте")
    type: Literal[ENTITY_TYPES]
    role: Literal[ROLES]
    sentiment: Literal[SENTIMENTS]
    normalized_id: str = Field(max_length=120, description="полное название в именительном падеже")


class Enrichment(BaseModel):
    """Результат обогащения одного документа.

    Порядок полей — это порядок генерации: при constrained decoding модель пишет
    их сверху вниз. Сначала пересказ и сущности, `confidence` последним — к этому
    моменту модель уже видит собственный разбор.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=400)
    entities: list[Entity] = Field(max_length=MAX_ENTITIES)
    topic: Literal[TOPICS]
    event_type: Literal[EVENT_TYPES]
    is_opinion: bool
    confidence: float = Field(ge=0.0, le=1.0)


JSON_SCHEMA = Enrichment.model_json_schema()
SCHEMA_VERSION = hashlib.sha256(
    json.dumps(JSON_SCHEMA, sort_keys=True, ensure_ascii=False).encode()
).hexdigest()[:8]


class FatalError(RuntimeError):
    """Ошибка, при которой повтор не нужен (схема отклонена, неверное имя модели)."""


def load_prompt(path=PROMPT_PATH):
    """Читает промпт и подставляет {topics} и {event_types} из TOPIC_HINTS / EVENT_TYPE_HINTS."""
    text = path.read_text(encoding="utf-8")
    for mark, hints in (("{topics}", TOPIC_HINTS), ("{event_types}", EVENT_TYPE_HINTS)):
        if mark not in text:
            raise SystemExit(f"в {path} нет подстановки {mark}")
        text = text.replace(mark, "\n".join(f"  {name} — {hint}" for name, hint in hints.items()))
    return text


def prompt_version(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def build_input(title, text, text_limit=TEXT_LIMIT):
    """Заголовок + первые text_limit символов текста."""
    return f"Заголовок: {title.strip()}\n\nТекст: {text.strip()[:text_limit]}"


def load_docs(path, limit=0):
    """Читает CSV, возвращает (все документы, первые limit).

    Обязательны title и text. article_id, story_id, source, dt_utc берутся если есть.
    Если article_id нет, он строится по номеру строки, как в embed.py и dedup_stories.ipynb.
    """
    df = pd.read_csv(path, dtype="string")
    missing = {"title", "text"} - set(df.columns)
    if missing:
        raise SystemExit(f"в {path} нет колонок: {sorted(missing)}")
    if "article_id" not in df.columns:
        df["article_id"] = [f"a_{i:07d}" for i in range(len(df))]
    for col in ("story_id", "source", "dt_utc"):
        if col not in df.columns:
            df[col] = pd.NA
    cols = ["article_id", "story_id", "source", "dt_utc", "title", "text"]
    docs = df[cols].fillna("").to_dict("records")
    return docs, (docs[:limit] if limit else docs)


class VLLM:
    """Клиент /v1/chat/completions, схема передаётся через response_format.

    guided_json в новых версиях vLLM игнорируется, поэтому response_format.
    requests.Session отдельная на каждый поток (thread-local).
    """

    def __init__(self, api, model, timeout, max_tokens=MAX_TOKENS, retries=RETRIES):
        self.api = api
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
        return session

    def _payload(self, system, user):
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "enrichment", "schema": JSON_SCHEMA, "strict": True},
            },
        }

    def chat(self, system, user):
        """Возвращает (content, completion_tokens). 4xx кроме 429 не повторяются."""
        for attempt in range(self.retries):
            try:
                resp = self._session().post(
                    self.api, json=self._payload(system, user), timeout=self.timeout
                )
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise FatalError(f"vLLM отверг запрос ({resp.status_code}): {resp.text[:600]}")
                resp.raise_for_status()
                data = resp.json()
                return (
                    data["choices"][0]["message"]["content"],
                    int(data.get("usage", {}).get("completion_tokens", 0)),
                )
            except FatalError:
                raise
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def probe(self, prompt, doc, text_limit=TEXT_LIMIT):
        """Пробный запрос перед прогоном, проверяет что сервер принимает схему."""
        raw, tokens = self.chat(prompt, build_input(doc["title"], doc["text"], text_limit))
        Enrichment.model_validate_json(raw)
        return tokens


def blank_row(doc):
    """Пустая строка результата с метаданными документа."""
    return {
        "article_id": doc.get("article_id", ""),
        "story_id": doc.get("story_id", ""),
        "source": doc.get("source", ""),
        "dt_utc": doc.get("dt_utc", ""),
        "summary": None,
        "entities": [],
        "topic": None,
        "event_type": None,
        "is_opinion": None,
        "confidence": None,
        "raw_response": "",
        "ok": False,
        "error": None,
        "completion_tokens": 0,
        "latency_s": 0.0,
    }


def enrich_one(client, prompt, doc, text_limit=TEXT_LIMIT):
    """Обогащение одного документа. При ошибке ok=False, наружу пробрасывается только FatalError.

    raw_response сохраняется и при ошибке разбора (обычно это обрыв по max_tokens).
    """
    row = blank_row(doc)
    started = time.perf_counter()
    try:
        raw, tokens = client.chat(prompt, build_input(doc["title"], doc["text"], text_limit))
        row["raw_response"] = raw
        row["completion_tokens"] = tokens
        row.update(Enrichment.model_validate_json(raw).model_dump())
        row["ok"] = True
    except FatalError:
        raise
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    row["latency_s"] = round(time.perf_counter() - started, 3)
    return row


def iter_rows(client, prompt, docs, concurrency, text_limit=TEXT_LIMIT):
    """Генератор результатов в порядке готовности.

    При выходе (Ctrl+C, FatalError) оставшиеся задачи отменяются.
    """
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = [pool.submit(enrich_one, client, prompt, doc, text_limit) for doc in docs]
        for future in as_completed(futures):
            yield future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_meta(args, prompt, extra=None):
    """Метаданные прогона: модель, версии схемы и промпта, параметры."""
    meta = {
        "model": args.model,
        "api": args.api,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": prompt_version(prompt),
        "prompt_path": str(args.prompt),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "text_limit": args.text_limit,
        "max_entities": MAX_ENTITIES,
        "concurrency": args.concurrency,
    }
    meta.update(extra or {})
    return meta


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def add_model_args(parser, api_default="http://127.0.0.1:8002/v1/chat/completions"):
    """Общие аргументы для P0 и P5. Порт 8002, т.к. на 8000 эмбеддер."""
    parser.add_argument("--api", default=api_default)
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--concurrency", type=int, default=32, help="запросов в полёте")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--text-limit", type=int, default=TEXT_LIMIT)
    parser.add_argument("--timeout", type=int, default=300)
    return parser
