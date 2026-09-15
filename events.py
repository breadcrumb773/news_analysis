"""Детекция событий: всплески в рядах сущностей и тем.

Ряд = число сюжетов с ключом в день / число сюжетов за день. Пока ряды строятся здесь
из шардов P5 (функции *_stub), а не берутся из trends.py.

Детекторы: z-score по скользящему окну (дни-выбросы) и Kleinberg (интервалы с весом).
Итоговый список - интервалы Kleinberg с числом дней z-score внутри. Топ всплесков сущностей
сверяется с configs/known_events_2016.csv: precision@20, recall@100.

Вход: data/enriched/shard_*.parquet, data/stories/canonical.csv (заголовки), configs/known_events_2016.csv.
Выход в data/events/: series.csv, zscore_days.csv, bursts.csv, top_bursts.csv.
Метрики data/reports/events_metrics.json, лог events_log.txt.

uv run python events.py
uv run python events.py --z-window 14 --z-threshold 2.5 --gamma 2
"""

import argparse
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
KNOWN_EVENTS = ROOT / "configs" / "known_events_2016.csv"

TOP_ENTITIES = 300  # самые частые сущности по числу сюжетов
Z_WINDOW = 28  # дней
Z_THRESHOLD = 3.0
MIN_STORIES = 5  # мин. сюжетов в день для выброса
BURST_S = 2.0  # множитель частоты в состоянии всплеска
BURST_GAMMA = 1.0  # цена перехода в всплеск, * ln(n)
MATCH_DAYS = 3  # допуск по дате при сверке с событиями
TOP_PRECISION = 20
TOP_RECALL = 100
HEADLINES = 3

log = logging.getLogger("events")


def setup_log(reports):
    """Лог в консоль и в data/reports/events_log.txt."""
    reports.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    file = logging.FileHandler(reports / "events_log.txt", encoding="utf-8")
    file.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.handlers = [file, console]
    log.setLevel(logging.INFO)


# TODO: заменить *_stub на чтение рядов trends.py
# doc_keys: article_id, story_id, date, kind, key
# series: kind, key, date, stories, day_total, share

def load_doc_keys_stub(folder):
    """Темы и сущности документов P5, возвращает (docs, doc_keys).

    Сущности по normalized_id без канонизации P6, разные формы одной сущности дают разные ряды.
    """
    paths = sorted(folder.glob("shard_*.parquet"))
    if not paths:
        raise SystemExit(f"в {folder} нет шардов shard_*.parquet, сначала запустите enrich.py (P5)")
    columns = ["article_id", "story_id", "dt_utc", "entities", "topic", "ok"]
    df = pd.concat([pq.read_table(p, columns=columns).to_pandas() for p in paths], ignore_index=True)
    df = df[df["ok"]].copy()
    # день по UTC
    df["date"] = pd.to_datetime(df["dt_utc"], utc=True, format="ISO8601").dt.tz_convert(None).dt.normalize()
    docs = df[["article_id", "story_id", "date"]]

    topics = df[["article_id", "story_id", "date", "topic"]].dropna(subset=["topic"])
    topics = topics.rename(columns={"topic": "key"}).assign(kind="topic")

    ents = df[["article_id", "story_id", "date", "entities"]].explode("entities")
    ents = ents.dropna(subset=["entities"])
    ents["key"] = [e["normalized_id"].strip() for e in ents["entities"]]
    ents = ents[ents["key"] != ""].drop(columns="entities").assign(kind="entity")
    ents = ents.drop_duplicates(["article_id", "key"])

    doc_keys = pd.concat([topics, ents], ignore_index=True)
    log.info(f"документов P5: {len(docs)}, пар документ-ключ: {len(doc_keys)}")
    return docs, doc_keys


def build_series_stub(docs, doc_keys):
    """Ряды по ключам и дням: число сюжетов и доля."""
    days = pd.date_range(docs["date"].min(), docs["date"].max(), freq="D")
    day_total = docs.groupby("date")["story_id"].nunique().reindex(days, fill_value=0)

    # все темы и TOP_ENTITIES сущностей
    n_stories = doc_keys.groupby(["kind", "key"])["story_id"].nunique()
    top_entities = n_stories.loc["entity"].nlargest(TOP_ENTITIES).index
    keep = (doc_keys["kind"] == "topic") | doc_keys["key"].isin(top_entities)
    counts = doc_keys[keep].groupby(["kind", "key", "date"])["story_id"].nunique()

    # заполняем пропущенные дни нулями
    table = counts.unstack("date", fill_value=0).reindex(columns=days, fill_value=0)
    series = table.reset_index().melt(id_vars=["kind", "key"], var_name="date", value_name="stories")
    series["day_total"] = series["date"].map(day_total)
    series["share"] = (series["stories"] / series["day_total"].replace(0, np.nan)).fillna(0.0)
    series = series.sort_values(["kind", "key", "date"]).reset_index(drop=True)
    log.info(f"рядов: {len(table)} ({(table.index.get_level_values('kind') == 'topic').sum()} тем), "
             f"дней: {len(days)} ({days[0].date()} — {days[-1].date()})")
    return series


def add_zscore(series, window, threshold):
    """z-score доли по предыдущим window дням (без текущего).

    std ограничен снизу долей одного сюжета в средний день, иначе у редких сущностей z огромный.
    """
    std_floor = 1.0 / series.loc[series["day_total"] > 0, "day_total"].mean()
    grouped = series.groupby(["kind", "key"])["share"]
    past_mean = grouped.transform(lambda x: x.shift(1).rolling(window).mean())
    past_std = grouped.transform(lambda x: x.shift(1).rolling(window).std())
    series["z"] = ((series["share"] - past_mean) / past_std.clip(lower=std_floor)).round(2)
    series["z_outlier"] = (series["z"] >= threshold) & (series["stories"] >= MIN_STORIES)
    return series


def kleinberg(r, d, s, gamma):
    """Kleinberg burst detection, 2 состояния, биномиальная модель, Витерби.

    r[t] - сюжетов с ключом в день t, d[t] - всего сюжетов.
    p0 = средняя доля, p1 = s * p0. Цена дня: -log правдоподобия (без биномиального коэффициента).
    Переход вверх gamma * ln(n), вниз 0.
    Возвращает [(start, end, weight)], weight = разница цен состояний 0 и 1 на интервале.
    """
    n = len(r)
    p0 = r.sum() / d.sum()
    p1 = min(s * p0, 0.9999)
    cost = np.vstack([-(r * np.log(p0) + (d - r) * np.log(1 - p0)),
                      -(r * np.log(p1) + (d - r) * np.log(1 - p1))])
    up = gamma * np.log(n)

    # total[i, t] - мин. цена пути до дня t в состоянии i
    total = np.zeros((2, n))
    came_from = np.zeros((2, n), dtype=int)
    total[0, 0] = cost[0, 0]
    total[1, 0] = cost[1, 0] + up
    for t in range(1, n):
        came_from[0, t] = 0 if total[0, t - 1] <= total[1, t - 1] else 1
        total[0, t] = total[came_from[0, t], t - 1] + cost[0, t]
        came_from[1, t] = 1 if total[1, t - 1] <= total[0, t - 1] + up else 0
        total[1, t] = min(total[1, t - 1], total[0, t - 1] + up) + cost[1, t]

    # обратный проход
    states = np.zeros(n, dtype=int)
    states[-1] = 0 if total[0, -1] <= total[1, -1] else 1
    for t in range(n - 1, 0, -1):
        states[t - 1] = came_from[states[t], t]

    # дни всплеска -> интервалы
    bursts = []
    t = 0
    while t < n:
        if states[t] == 1:
            start = t
            while t + 1 < n and states[t + 1] == 1:
                t += 1
            weight = float((cost[0, start:t + 1] - cost[1, start:t + 1]).sum())
            bursts.append((start, t, weight))
        t += 1
    return bursts


def find_bursts(series, s, gamma):
    """Всплески Kleinberg по всем рядам с пиком и числом дней z-score."""
    rows = []
    for (kind, key), part in series.groupby(["kind", "key"], sort=False):
        r = part["stories"].to_numpy()
        d = part["day_total"].to_numpy()
        if r.sum() == 0:
            continue
        for start, end, weight in kleinberg(r, d, s, gamma):
            chunk = part.iloc[start:end + 1]
            peak = chunk.loc[chunk["stories"].idxmax()]
            rows.append({
                "kind": kind,
                "key": key,
                "start": chunk["date"].iloc[0].date(),
                "end": chunk["date"].iloc[-1].date(),
                "days": len(chunk),
                "weight": round(weight, 1),
                "stories": int(chunk["stories"].sum()),
                "peak_date": peak["date"].date(),
                "peak_stories": int(peak["stories"]),
                "max_z": chunk["z"].max(),
                "z_days": int(chunk["z_outlier"].sum()),
            })
    return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)


def load_known_events(path):
    """Список событий: start, end, event, keys (через ;)."""
    events = pd.read_csv(path, comment="#")
    events["start"] = pd.to_datetime(events["start"]).dt.date
    events["end"] = pd.to_datetime(events["end"]).dt.date
    events["words"] = events["keys"].str.lower().str.replace("ё", "е").str.split(";")
    log.info(f"реальных событий в списке: {len(events)}")
    return events


def match_event(key, peak_date, events):
    """Событие, у которого даты рядом с пиком и слово из keys входит в ключ как подстрока."""
    label = key.lower().replace("ё", "е")
    margin = timedelta(days=MATCH_DAYS)
    for ev in events.itertuples():
        near = ev.start - margin <= peak_date <= ev.end + margin
        if near and any(word in label for word in ev.words):
            return ev.event
    return None


def peak_headlines(doc_keys, titles, kind, key, date):
    """Заголовки пикового дня."""
    mask = (doc_keys["kind"] == kind) & (doc_keys["key"] == key) & (doc_keys["date"] == pd.Timestamp(date))
    ids = doc_keys.loc[mask, "article_id"].head(HEADLINES)
    return " | ".join(titles.reindex(ids).dropna())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--events", type=Path, default=KNOWN_EVENTS)
    ap.add_argument("--z-window", type=int, default=Z_WINDOW)
    ap.add_argument("--z-threshold", type=float, default=Z_THRESHOLD)
    ap.add_argument("--s", type=float, default=BURST_S)
    ap.add_argument("--gamma", type=float, default=BURST_GAMMA)
    args = ap.parse_args()

    out = args.data / "events"
    reports = args.data / "reports"
    out.mkdir(parents=True, exist_ok=True)
    setup_log(reports)
    log.info(f"=== события: z_window={args.z_window}, z_threshold={args.z_threshold}, "
             f"s={args.s}, gamma={args.gamma}")
    started = time.time()

    # ряды
    docs, doc_keys = load_doc_keys_stub(args.data / "enriched")
    series = build_series_stub(docs, doc_keys)

    # z-score
    series = add_zscore(series, args.z_window, args.z_threshold)
    series.drop(columns="z_outlier").to_csv(out / "series.csv", index=False)
    z_days = series[series["z_outlier"]].drop(columns="z_outlier").sort_values("z", ascending=False)
    z_days.to_csv(out / "zscore_days.csv", index=False)
    log.info(f"z-score: {len(z_days)} дней-выбросов в {z_days.groupby(['kind', 'key']).ngroups} рядах")

    # Kleinberg
    step = time.time()
    bursts = find_bursts(series, args.s, args.gamma)
    bursts.to_csv(out / "bursts.csv", index=False)
    log.info(f"Kleinberg: {len(bursts)} всплесков "
             f"({(bursts['kind'] == 'entity').sum()} сущностей, {(bursts['kind'] == 'topic').sum()} тем), "
             f"{time.time() - step:.0f} с")
    # темы только в лог, со списком событий не сверяются
    for row in bursts[bursts["kind"] == "topic"].head(10).itertuples():
        log.info(f"  тема  {row.start} - {row.end}  вес {row.weight:>8}  {row.key}")

    # сверка топа сущностей со списком событий
    events = load_known_events(args.events)
    titles = pd.read_csv(args.data / "stories" / "canonical.csv", usecols=["article_id", "title"],
                         dtype="string").set_index("article_id")["title"]
    top = bursts[bursts["kind"] == "entity"].head(TOP_RECALL).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top["what_it_was"] = [match_event(k, p, events) for k, p in zip(top["key"], top["peak_date"])]
    top["headlines"] = [peak_headlines(doc_keys, titles, "entity", k, p)
                        for k, p in zip(top["key"], top["peak_date"])]
    # для ручной разметки
    top["manual_check"] = ""
    top.to_csv(out / "top_bursts.csv", index=False)

    head = top.head(TOP_PRECISION)
    precision = float(head["what_it_was"].notna().mean())
    found = top["what_it_was"].dropna().unique()
    recall = len(found) / len(events)
    log.info(f"precision@{TOP_PRECISION} = {precision:.2f}, recall@{TOP_RECALL} = {recall:.2f} "
             f"({len(found)} из {len(events)} событий)")
    for row in head.itertuples():
        log.info(f"  {row.rank:>3}. {row.peak_date}  вес {row.weight:>8}  {row.key:<30} -> "
                 f"{row.what_it_was or '?'}")

    # доля топ-всплесков с подтверждением z-score и распределение пиков по дням недели
    agreement = float((head["z_days"] > 0).mean())
    weekday = pd.to_datetime(bursts.loc[bursts["kind"] == "entity", "peak_date"]).dt.day_name()
    weekday_share = weekday.value_counts(normalize=True).round(3).to_dict()
    log.info(f"z-score подтверждает {agreement:.0%} топ-{TOP_PRECISION} всплесков; "
             f"пики по дням недели: {weekday_share}")

    metrics = {
        "series": int(series.groupby(["kind", "key"]).ngroups),
        "days": int(series["date"].nunique()),
        "params": {
            "top_entities": TOP_ENTITIES, "z_window": args.z_window, "z_threshold": args.z_threshold,
            "min_stories": MIN_STORIES, "s": args.s, "gamma": args.gamma, "match_days": MATCH_DAYS,
        },
        "zscore_days": int(len(z_days)),
        "bursts": int(len(bursts)),
        "bursts_entity": int((bursts["kind"] == "entity").sum()),
        "bursts_topic": int((bursts["kind"] == "topic").sum()),
        "known_events": int(len(events)),
        f"precision_at_{TOP_PRECISION}": round(precision, 3),
        f"recall_at_{TOP_RECALL}": round(recall, 3),
        "events_found": sorted(found.tolist()),
        f"zscore_agreement_top_{TOP_PRECISION}": round(agreement, 3),
        "peak_weekday_share": weekday_share,
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }
    (reports / "events_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info(f"готово за {metrics['elapsed_minutes']} мин, результаты в {out}")


if __name__ == "__main__":
    main()
