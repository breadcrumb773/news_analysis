"""Тренды: временные ряды по сюжетам с нормировкой на дневной объём.

Считаем по сюжетам, а не статьям. Сюжет активен в день, если в этот день по нему была статья
(story_map). Метки (topic, event_type) берутся из обогащения канонического документа, сущности
и тональность из mentions.parquet. Доля = сюжеты с меткой / обогащённые сюжеты дня. Дни по Москве.

Считается: ряды тема/тип события/сущность/тональность по дням, время жизни сюжетов,
профиль по дням недели и месяцам, тональность к топ-сущностям по неделям.

Вход: data/stories/story_map.csv (P4), data/enriched/shard_*.parquet (P5), data/entities/mentions.parquet (P6).
Выход: data/trends/*.csv, в data/reports/ trends_log.txt, trends_summary.json, trends_*.csv, trends_*.png.
В дневных рядах только ненулевые дни.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

TZ = "Europe/Moscow"
SENTIMENTS = ["pos", "neu", "neg", "unknown"]
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

MIN_STORIES = 5  # мин. сюжетов у сущности для рядов
TOP_ENTITIES = 20  # сущностей для недельной тональности
MIN_WEEK_STORIES = 5  # мин. сюжетов в неделю для точки на графике
TOP_TOPICS = 8
LONGEST = 20

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#898781"
GRID = "#e6e5e1"

log = logging.getLogger("trends")


def setup_log(reports):
    """Лог в консоль и в data/reports/trends_log.txt."""
    reports.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    file = logging.FileHandler(reports / "trends_log.txt", encoding="utf-8")
    file.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.handlers = [file, console]
    log.setLevel(logging.INFO)


def load_story_map(path):
    """Все статьи с story_id и датой по Москве."""
    sm = pd.read_csv(path, usecols=["article_id", "source", "dt_utc", "story_id"])
    sm["dt"] = pd.to_datetime(sm["dt_utc"], utc=True, format="ISO8601").dt.tz_convert(TZ)
    sm["date"] = sm["dt"].dt.tz_localize(None).dt.normalize()
    log.info(f"статей: {len(sm)}, сюжетов: {sm['story_id'].nunique()}, "
             f"период {sm['date'].min():%Y-%m-%d} — {sm['date'].max():%Y-%m-%d}")
    return sm


def load_story_labels(folder):
    """topic и event_type сюжета из P5."""
    paths = sorted(folder.glob("shard_*.parquet"))
    if not paths:
        raise SystemExit(f"в {folder} нет шардов shard_*.parquet, сначала запустите enrich.py (P5)")
    df = pd.concat([pd.read_parquet(p, columns=["story_id", "topic", "event_type", "ok"])
                    for p in paths], ignore_index=True)
    df = df[df["ok"]].drop_duplicates("story_id")
    log.info(f"обогащённых сюжетов: {len(df)}")
    return df.set_index("story_id")[["topic", "event_type"]]


def load_mentions(path):
    """Упоминания из P6."""
    if not path.exists():
        raise SystemExit(f"нет {path}, сначала запустите postprocess.py (P6)")
    mentions = pd.read_parquet(path, columns=["story_id", "entity_id", "canonical", "type", "sentiment"])
    log.info(f"упоминаний: {len(mentions)}, сущностей: {mentions['entity_id'].nunique()}")
    return mentions


def daily_volume(sm, story_days, labels):
    """Объём по дням: статьи (всего и по источникам), сюжеты, новые сюжеты.

    enriched_stories используется как знаменатель долей.
    """
    days = pd.date_range(sm["date"].min(), sm["date"].max(), freq="D", name="date")
    volume = pd.DataFrame(index=days)
    volume["articles"] = sm.groupby("date").size()
    volume["stories"] = story_days.groupby("date").size()
    volume["new_stories"] = story_days.groupby("story_id")["date"].min().value_counts()
    enriched = story_days[story_days["story_id"].isin(labels.index)]
    volume["enriched_stories"] = enriched.groupby("date").size()
    by_source = sm.groupby(["date", "source"]).size().unstack(fill_value=0)
    volume = volume.join(by_source.add_prefix("articles_"))
    volume = volume.fillna(0).astype(int)
    volume.insert(0, "weekday", volume.index.dayofweek)  # 0 = пн
    volume.insert(1, "month", volume.index.month)
    return volume


def daily_share(story_days, labels, key, volume):
    """Число и доля сюжетов дня с меткой key.

    labels: story_id -> key, у сюжета может быть несколько строк. Пара (сюжет, метка) считается один раз.
    """
    pairs = labels[["story_id", key]].drop_duplicates()
    part = story_days[["story_id", "date"]].merge(pairs, on="story_id")
    series = part.groupby(["date", key]).size().rename("n_stories").reset_index()
    series["share"] = (series["n_stories"] / series["date"].map(volume["enriched_stories"])).round(5)
    return series


def sentiment_daily(story_days, mentions):
    """Тональность по дням, по уникальным (сюжет, сущность, тональность)."""
    pairs = mentions[["story_id", "entity_id", "sentiment"]].drop_duplicates()
    part = story_days[["story_id", "date"]].merge(pairs, on="story_id")
    table = part.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    table = table.reindex(columns=SENTIMENTS, fill_value=0)
    table.columns.name = None
    shares = table.div(table.sum(axis=1), axis=0).round(4).add_suffix("_share")
    return table.join(shares)


def entity_sentiment_weekly(story_days, mentions, top_ids):
    """Тональность к топ-сущностям по неделям (по дням слишком мало данных).

    sentiment_index = (pos - neg) / (pos + neu + neg), unknown не учитывается.
    """
    pairs = mentions.loc[mentions["entity_id"].isin(top_ids),
                         ["story_id", "entity_id", "canonical", "sentiment"]].drop_duplicates()
    part = story_days[["story_id", "date"]].merge(pairs, on="story_id")
    part["week"] = part["date"].dt.to_period("W").dt.start_time

    keys = ["entity_id", "canonical", "week"]
    table = part.groupby(keys + ["sentiment"]).size().unstack(fill_value=0)
    table = table.reindex(columns=SENTIMENTS, fill_value=0)
    table.columns.name = None
    table["n_stories"] = part.groupby(keys)["story_id"].nunique()
    known = table["pos"] + table["neu"] + table["neg"]
    table["sentiment_index"] = ((table["pos"] - table["neg"]) / known.where(known > 0)).round(3)
    return table.reset_index()


def story_lifetime(sm, labels):
    """Время жизни сюжета от первой до последней статьи.

    lifetime_days в календарных днях включительно (один день = 1).
    """
    g = sm.groupby("story_id")
    life = pd.DataFrame({
        "size": g.size(),
        "n_sources": g["source"].nunique(),
        "first_dt": g["dt"].min(),
        "last_dt": g["dt"].max(),
    })
    life["lifetime_days"] = (life["last_dt"].dt.normalize() - life["first_dt"].dt.normalize()).dt.days + 1
    life["lifetime_hours"] = ((life["last_dt"] - life["first_dt"]).dt.total_seconds() / 3600).round(1)
    life["topic"] = life.index.map(labels["topic"])
    return life


def lifetime_report(life, mentions):
    """Распределение времени жизни по сюжетам из 2+ статей, квантили, самые долгие."""
    multi = life[life["size"] > 1]
    dist = multi["lifetime_days"].value_counts().sort_index().rename("stories").to_frame()
    dist["share"] = (dist["stories"] / len(multi)).round(4)
    dist["cum_share"] = dist["share"].cumsum().round(4)
    dist.index.name = "lifetime_days"

    stats = {
        "stories": len(life),
        "single_article_stories": int((life["size"] == 1).sum()),
        "multi_article_stories": len(multi),
        "multi_days_median": float(multi["lifetime_days"].median()),
        "multi_days_p90": float(multi["lifetime_days"].quantile(0.9)),
        "multi_days_p99": float(multi["lifetime_days"].quantile(0.99)),
        "multi_days_max": int(multi["lifetime_days"].max()),
        "multi_hours_median": round(float(multi["lifetime_hours"].median()), 1),
        "multi_hours_p90": round(float(multi["lifetime_hours"].quantile(0.9)), 1),
        "multi_share_longer_3_days": round(float((multi["lifetime_days"] > 3).mean()), 4),
        "multi_share_longer_7_days": round(float((multi["lifetime_days"] > 7).mean()), 4),
    }

    # для самых долгих сюжетов топ-3 сущности вместо заголовка
    longest = multi.sort_values(["lifetime_days", "size"], ascending=False).head(LONGEST).copy()
    names = mentions[mentions["story_id"].isin(longest.index)].groupby("story_id")["canonical"] \
        .agg(lambda s: ", ".join(s.value_counts().head(3).index))
    longest["top_entities"] = names
    return dist, stats, longest


def volume_profile(volume, by):
    """Средний объём по дням недели или месяцам и отношение к среднему."""
    columns = ["articles", "stories", "new_stories", "enriched_stories"]
    profile = volume.groupby(by)[columns].mean().round(1)
    for col in ["articles", "stories"]:
        profile[f"{col}_vs_mean"] = (profile[col] / volume[col].mean()).round(3)
    return profile


def label_lift(series, volume, key, by):
    """Отношение доли метки в день недели (месяц) к её доле за год."""
    groups = series["date"].map(volume[by]).rename(by)
    counts = series.groupby([series[key], groups])["n_stories"].sum().unstack(fill_value=0)
    shares = counts / volume.groupby(by)["enriched_stories"].sum()
    year_share = counts.sum(axis=1) / volume["enriched_stories"].sum()
    lift = shares.div(year_share, axis=0).round(2)
    lift.insert(0, "year_share", year_share.round(4))
    return lift.sort_values("year_share", ascending=False)


def save_fig(fig, axes, path):
    """Оформление осей и сохранение."""
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_volume(volume, path):
    """Статьи и сюжеты по дням + скользящее среднее за 7 дней."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for col, label, color in [("articles", "статьи", COLORS[0]), ("stories", "сюжеты", COLORS[1])]:
        ax.plot(volume.index, volume[col], color=color, linewidth=0.8, alpha=0.35)
        ax.plot(volume.index, volume[col].rolling(7, center=True).mean(), color=color,
                linewidth=2, label=f"{label} (среднее за 7 дней)")
    ax.set_title("Объём корпуса по дням")
    ax.set_ylabel("в день")
    ax.legend(frameon=False)
    save_fig(fig, [ax], path)


def plot_weekday(profile, path):
    """Объём по дням недели относительно среднего."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(profile))
    ax.bar([i - 0.2 for i in x], profile["articles_vs_mean"], width=0.38, color=COLORS[0], label="статьи")
    ax.bar([i + 0.2 for i in x], profile["stories_vs_mean"], width=0.38, color=COLORS[1], label="сюжеты")
    ax.axhline(1.0, color=GRAY, linewidth=1)
    ax.set_xticks(list(x), [WEEKDAYS[d] for d in profile.index])
    ax.set_title("День недели: объём относительно среднего дня")
    ax.legend(frameon=False)
    save_fig(fig, [ax], path)


def plot_lifetime(dist, path):
    """Гистограмма времени жизни, лог. шкала."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dist.index, dist["stories"], color=COLORS[0], width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("время жизни, дней")
    ax.set_ylabel("сюжетов")
    ax.set_title("Время жизни сюжетов из двух и более статей")
    save_fig(fig, [ax], path)


def plot_topics_weekly(topics, volume, path):
    """Недельная доля крупнейших тем."""
    weekly = topics.groupby([pd.Grouper(key="date", freq="W"), "topic"])["n_stories"].sum() \
        .unstack(fill_value=0)
    share = weekly.div(volume["enriched_stories"].resample("W").sum(), axis=0)
    top = weekly.sum().nlargest(TOP_TOPICS).index

    fig, ax = plt.subplots(figsize=(12, 5))
    for topic, color in zip(top, COLORS):
        ax.plot(share.index, share[topic] * 100, color=color, linewidth=2, label=topic)
    ax.set_title("Доля темы среди сюжетов недели")
    ax.set_ylabel("% сюжетов")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save_fig(fig, [ax], path)


def plot_entity_sentiment(weekly, top_ids, path, n=6):
    """sentiment_index по неделям для n сущностей."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, entity_id in zip(axes.flat, top_ids[:n]):
        part = weekly[(weekly["entity_id"] == entity_id) & (weekly["n_stories"] >= MIN_WEEK_STORIES)]
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.plot(part["week"], part["sentiment_index"], color=COLORS[0], linewidth=2, marker="o", markersize=3)
        name = weekly.loc[weekly["entity_id"] == entity_id, "canonical"].iloc[0]
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-1, 1)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.suptitle("Тональность к сущности по неделям: (pos − neg) / (pos + neu + neg)")
    save_fig(fig, list(axes.flat), path)


def main():
    ap = argparse.ArgumentParser(description="Анализ трендов")
    ap.add_argument("--data", default=DATA, help="корень data/")
    args = ap.parse_args()

    data = Path(args.data)
    out = data / "trends"
    reports = data / "reports"
    setup_log(reports)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log.info("=" * 70)
    log.info(f"старт: data={data}")

    sm = load_story_map(data / "stories" / "story_map.csv")
    labels = load_story_labels(data / "enriched")
    mentions = load_mentions(data / "entities" / "mentions.parquet")

    # пары (сюжет, день) с хотя бы одной статьёй
    story_days = sm.groupby(["story_id", "date"]).size().rename("articles").reset_index()
    log.info(f"пар «сюжет, день»: {len(story_days)}")

    volume = daily_volume(sm, story_days, labels)
    volume.to_csv(out / "daily_volume.csv")

    # ряды
    story_labels = labels.reset_index()
    topics = daily_share(story_days, story_labels, "topic", volume)
    topics.to_csv(out / "topic_daily.csv", index=False)
    events = daily_share(story_days, story_labels, "event_type", volume)
    events.to_csv(out / "event_type_daily.csv", index=False)

    # сущности с >= MIN_STORIES сюжетов
    n_stories = mentions.groupby("entity_id")["story_id"].nunique().sort_values(ascending=False)
    frequent = n_stories[n_stories >= MIN_STORIES].index
    entities = daily_share(story_days, mentions[mentions["entity_id"].isin(frequent)], "entity_id", volume)
    names = mentions.drop_duplicates("entity_id").set_index("entity_id")
    entities.insert(2, "canonical", entities["entity_id"].map(names["canonical"]))
    entities.insert(3, "type", entities["entity_id"].map(names["type"]))
    entities.to_csv(out / "entity_daily.csv", index=False)
    log.info(f"ряды: тем {topics['topic'].nunique()}, типов событий {events['event_type'].nunique()}, "
             f"сущностей {len(frequent)} (из {len(n_stories)}, порог {MIN_STORIES} сюжетов)")

    sentiment = sentiment_daily(story_days, mentions)
    sentiment.to_csv(out / "sentiment_daily.csv")

    top_ids = list(n_stories.index[:TOP_ENTITIES])
    entity_weekly = entity_sentiment_weekly(story_days, mentions, top_ids)
    entity_weekly.to_csv(out / "entity_sentiment_weekly.csv", index=False)

    life = story_lifetime(sm, labels)
    life.to_csv(out / "story_lifetime.csv")
    dist, life_stats, longest = lifetime_report(life, mentions)
    dist.to_csv(reports / "trends_lifetime_dist.csv")
    longest.to_csv(reports / "trends_longest_stories.csv")
    log.info(f"время жизни (сюжеты из 2+ статей): медиана {life_stats['multi_days_median']} дн., "
             f"p90 {life_stats['multi_days_p90']}, p99 {life_stats['multi_days_p99']}, "
             f"максимум {life_stats['multi_days_max']}; дольше недели "
             f"{life_stats['multi_share_longer_7_days']:.1%}")

    # профиль по дням недели и месяцам (данные за один год, сезонность от тренда не отделить)
    weekday = volume_profile(volume, "weekday")
    weekday.index = [WEEKDAYS[d] for d in weekday.index]
    weekday.index.name = "weekday"
    weekday.to_csv(reports / "trends_weekday.csv")
    volume_profile(volume, "month").to_csv(reports / "trends_monthly.csv")

    topic_weekday = label_lift(topics, volume, "topic", "weekday")
    topic_weekday = topic_weekday.rename(columns=dict(enumerate(WEEKDAYS)))
    topic_weekday.to_csv(reports / "trends_topic_weekday.csv")
    label_lift(topics, volume, "topic", "month").to_csv(reports / "trends_topic_monthly.csv")

    weekend = volume[volume["weekday"] >= 5]
    workdays = volume[volume["weekday"] < 5]
    weekend_ratio = round(float(weekend["articles"].mean() / workdays["articles"].mean()), 3)
    log.info(f"выходной день / будний по статьям: {weekend_ratio}")
    log.info("\n" + weekday.to_string())

    plot_volume(volume, reports / "trends_volume.png")
    plot_weekday(weekday.set_axis(range(7)), reports / "trends_weekday.png")
    plot_lifetime(dist, reports / "trends_lifetime.png")
    plot_topics_weekly(topics, volume, reports / "trends_topics_weekly.png")
    plot_entity_sentiment(entity_weekly, top_ids, reports / "trends_entity_sentiment.png")

    summary = {
        "timezone": TZ,
        "days": len(volume),
        "articles": len(sm),
        "enriched_stories": len(labels),
        "story_days": len(story_days),
        "articles_per_day_mean": round(float(volume["articles"].mean()), 1),
        "stories_per_day_mean": round(float(volume["stories"].mean()), 1),
        "weekend_to_workday_articles": weekend_ratio,
        "weekend_to_workday_stories": round(float(weekend["stories"].mean() / workdays["stories"].mean()), 3),
        "lifetime": life_stats,
        "series": {
            "topics": int(topics["topic"].nunique()),
            "event_types": int(events["event_type"].nunique()),
            "entities": len(frequent),
            "entity_min_stories": MIN_STORIES,
        },
        "top_entities": [f"{names.loc[e, 'canonical']} ({n_stories[e]})" for e in top_ids],
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    (reports / "trends_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    log.info(f"ряды: {out}; отчёты и графики: {reports}/trends_*")
    log.info(f"готово за {summary['elapsed_minutes']} мин")


if __name__ == "__main__":
    main()
