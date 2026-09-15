"""Загрузка данных и шаг 1: упоминания сущностей и частотный словарь форм."""

import pandas as pd
import pyarrow.parquet as pq

from enrich_schema import TEXT_LIMIT
from postprocess.common import log

ENRICHED_COLUMNS = ["article_id", "story_id", "source", "dt_utc", "entities", "confidence", "ok"]


def load_enriched(folder):
    """Читает шарды P5, оставляет строки с ok=True.

    raw_response не читаем, он большой. Недописанные parts/*.jsonl не берутся.
    """
    paths = sorted(folder.glob("shard_*.parquet"))
    if not paths:
        raise SystemExit(f"в {folder} нет шардов shard_*.parquet, сначала запустите enrich.py (P5)")
    df = pd.concat([pq.read_table(p, columns=ENRICHED_COLUMNS).to_pandas() for p in paths],
                   ignore_index=True)
    log.info(f"обогащение: {len(paths)} шардов, {len(df)} документов, из них ok: {df['ok'].sum()}")
    return df[df["ok"]].reset_index(drop=True)


def load_articles(path):
    """Канонические документы из P4.

    Возвращает таблицу article_id -> (title, story_size) и словарь article_id -> текст,
    который видела модель в P5 (заголовок + первые TEXT_LIMIT символов). Словарь нужен для grounded.
    """
    df = pd.read_csv(path, usecols=["article_id", "title", "text", "story_size"],
                     dtype={"article_id": "string", "title": "string", "text": "string"})
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    inputs = dict(zip(df["article_id"], df["title"] + " " + df["text"].str.slice(0, TEXT_LIMIT)))
    articles = df[["article_id", "title", "story_size"]].set_index("article_id")
    log.info(f"канонических документов: {len(articles)}")
    return articles, inputs


def explode_mentions(enriched, inputs):
    """Одна строка на упоминание. confidence берётся от документа."""
    rows = []
    for doc in enriched.itertuples(index=False):
        text = inputs.get(doc.article_id, "")
        for e in doc.entities:
            rows.append((doc.article_id, doc.story_id, doc.dt_utc, doc.source,
                         e["name"], e["type"], e["role"], e["sentiment"], doc.confidence,
                         e["name"] in text, e["normalized_id"]))
    mentions = pd.DataFrame(rows, columns=[
        "article_id", "story_id", "date", "source", "name", "type", "role", "sentiment",
        "confidence", "grounded", "normalized_llm"])
    mentions["date"] = pd.to_datetime(mentions["date"], utc=True, format="ISO8601").dt.date
    log.info(f"упоминаний: {len(mentions)}, найдены в тексте: {mentions['grounded'].mean():.1%}")
    return mentions


def llm_norm_mode(mentions, keys):
    """Мода normalized_llm по группе и её доля."""
    counts = mentions.groupby(keys + ["normalized_llm"]).size().rename("n").reset_index()
    counts = counts.sort_values("n", ascending=False, kind="stable").drop_duplicates(keys)
    total = mentions.groupby(keys).size().rename("total").reset_index()
    counts = counts.merge(total, on=keys)
    counts["llm_norm_agree"] = (counts["n"] / counts["total"]).round(3)
    return counts[keys + ["normalized_llm", "llm_norm_agree"]]


def sample_titles(mentions, keys, articles):
    """Заголовок из самого большого сюжета с этой формой."""
    part = mentions[keys + ["article_id"]].copy()
    part["story_size"] = part["article_id"].map(articles["story_size"]).fillna(0)
    part = part.sort_values("story_size", ascending=False, kind="stable").drop_duplicates(keys)
    part["sample_title"] = part["article_id"].map(articles["title"])
    return part[keys + ["sample_title"]]


def surface_forms(mentions, articles):
    """Шаг 1. Частотный словарь по парам (name, type).

    Основная частота n_stories. В P5 брался один документ на сюжет, поэтому n_docs ~ n_stories.
    """
    keys = ["name", "type"]
    forms = mentions.groupby(keys).agg(
        n_mentions=("article_id", "size"),
        n_docs=("article_id", "nunique"),
        n_stories=("story_id", "nunique"),
        n_sources=("source", "nunique"),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
        conf_mean=("confidence", "mean"),
    ).reset_index()
    forms["conf_mean"] = forms["conf_mean"].round(3)

    mode = llm_norm_mode(mentions, keys).rename(columns={"normalized_llm": "llm_norm_mode"})
    forms = forms.merge(mode, on=keys, how="left")
    forms = forms.merge(sample_titles(mentions, keys, articles), on=keys, how="left")
    forms = forms.sort_values(["n_stories", "n_mentions", "name"],
                              ascending=[False, False, True]).reset_index(drop=True)
    log.info(f"шаг 1: поверхностных форм {len(forms)}")
    return forms
