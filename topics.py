"""Темы: BERTopic на эмбеддингах из P2 по каноническим документам P4 (один на сюжет).

Эмбеддинги P2 -> UMAP (5 измерений) -> HDBSCAN -> c-TF-IDF по леммам -> слияние до NR_TOPICS.
Для c-TF-IDF берутся леммы существительных и прилагательных (без стоп-листа).
topic из P5 используется для NMI/ARI.

Вход: data/stories/canonical.csv, data/embeddings/matrix.npy + id_map.csv, data/enriched/shard_*.parquet.
Выход в data/topics/: topics.csv, doc_topics.csv, topics_over_time.csv/.html (по неделям),
hierarchy.csv, lemmas.parquet (кэш). Метрики в data/reports/topics_metrics.json, лог topics_log.txt.

uv run python topics.py
uv run python topics.py --min-cluster-size 100 --nr-topics 80
Подробнее docs/run_topics.md.
"""

import argparse
import json
import logging
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pymorphy3
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
from umap import UMAP

DATA = Path(__file__).resolve().parent.parent / "data"

TEXT_LIMIT = 1500  # как в P2 и P5
SEED = 42  # с seed UMAP однопоточный
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 5
MIN_CLUSTER_SIZE = 50
NR_TOPICS = 100
TOP_WORDS = 10
TOT_TOP_TOPICS = 10  # тем на графике topics_over_time

# существительные и полные прилагательные, местоименные (Apro) отбрасываются в lemma()
KEEP_POS = {"NOUN", "ADJF"}
WORD = re.compile(r"[а-яё]+(?:-[а-яё]+)*")

log = logging.getLogger("topics")
morph = pymorphy3.MorphAnalyzer()


def setup_log(reports):
    """Лог в консоль и в data/reports/topics_log.txt."""
    reports.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    file = logging.FileHandler(reports / "topics_log.txt", encoding="utf-8")
    file.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.handlers = [file, console]
    log.setLevel(logging.INFO)


def load_canonical(path):
    """Канонические документы P4."""
    df = pd.read_csv(path, usecols=["article_id", "dt_utc", "title", "text"], dtype="string")
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, format="ISO8601")
    log.info(f"канонических документов: {len(df)}")
    return df.reset_index(drop=True)


def load_embeddings(folder, article_ids):
    """Векторы из матрицы P2 в порядке article_ids."""
    id_map = pd.read_csv(folder / "id_map.csv", usecols=["row", "article_id"],
                         dtype={"article_id": "string"})
    row_of = pd.Series(id_map["row"].to_numpy(), index=id_map["article_id"])
    rows = row_of.reindex(article_ids)
    if rows.isna().any():
        raise SystemExit(f"{rows.isna().sum()} статей нет в id_map.csv, матрица не соответствует P4")

    # mmap, чтобы не читать всю матрицу
    matrix = np.load(folder / "matrix.npy", mmap_mode="r")
    emb = np.asarray(matrix[rows.to_numpy(dtype=np.int64)], dtype=np.float32)
    log.info(f"эмбеддинги: {emb.shape[0]} x {emb.shape[1]}")
    return emb


def load_p5_topics(folder):
    """topic из P5 (только ok=True)."""
    paths = sorted(folder.glob("shard_*.parquet"))
    if not paths:
        raise SystemExit(f"в {folder} нет шардов shard_*.parquet, сначала запустите enrich.py (P5)")
    df = pd.concat([pq.read_table(p, columns=["article_id", "topic", "ok"]).to_pandas()
                    for p in paths], ignore_index=True)
    df = df[df["ok"] & df["topic"].notna()]
    log.info(f"topic из P5: {len(df)} документов")
    return df.set_index("article_id")["topic"]


@lru_cache(maxsize=None)
def lemma(word):
    """Лемма для NOUN/ADJF длиной от 3 символов, иначе None."""
    parse = morph.parse(word)[0]
    if parse.tag.POS not in KEEP_POS or "Apro" in parse.tag:
        return None
    normal = parse.normal_form.replace("ё", "е")
    return normal if len(normal) >= 3 else None


def lemmatize(text):
    words = (lemma(w) for w in WORD.findall(text.lower()))
    return " ".join(w for w in words if w)


def load_or_build_lemmas(df, cache):
    """Леммы по документам, с кэшем в parquet (проверка по списку article_id)."""
    if cache.exists():
        cached = pd.read_parquet(cache)
        if cached["article_id"].tolist() == df["article_id"].tolist():
            log.info(f"леммы из кэша: {cache}")
            return cached["lemmas"].tolist()
        log.info("кэш лемм собран на других статьях, пересчитываем")

    started = time.time()
    texts = df["title"] + " " + df["text"].str.slice(0, TEXT_LIMIT)
    lemmas = [lemmatize(t) for t in tqdm(texts, desc="леммы", unit="док")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"article_id": df["article_id"], "lemmas": lemmas}).to_parquet(cache, index=False)
    log.info(f"лемматизация: {time.time() - started:.0f} с, "
             f"разных слов в кэше pymorphy3: {lemma.cache_info().currsize}")
    return lemmas


def build_model(min_cluster_size):
    umap_model = UMAP(n_neighbors=UMAP_NEIGHBORS, n_components=UMAP_COMPONENTS,
                      min_dist=0.0, metric="cosine", random_state=SEED)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                            cluster_selection_method="eom")
    # на входе леммы через пробел. min_df не задаём, в BERTopic векторайзер работает по темам, а не документам
    vectorizer = CountVectorizer(token_pattern=r"\S+", lowercase=False)
    # reduce_frequent_words понижает вес общих слов (год, россия)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    return BERTopic(
        embedding_model=None,  # эмбеддинги из P2
        # с english (по умолчанию) BERTopic оставляет только [A-Za-z0-9]
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        top_n_words=TOP_WORDS,
        calculate_probabilities=False,  # probabilities_ из HDBSCAN
        verbose=True,
    )


def top_words(model, topic):
    # убираем пустые слова, которыми BERTopic дополняет короткие темы
    return [w for w, _ in model.get_topic(topic)[:TOP_WORDS] if w]


def coherence_cv(model, topic_ids, lemmas):
    """Coherence c_v по темам и среднее, на леммах."""
    texts = [doc.split() for doc in lemmas]
    dictionary = Dictionary(texts)
    words = [top_words(model, t) for t in topic_ids]
    cm = CoherenceModel(topics=words, texts=texts, dictionary=dictionary,
                        coherence="c_v", topn=TOP_WORDS,
                        processes=1)  # с пулом процессов медленнее
    per_topic = cm.get_coherence_per_topic()
    return per_topic, float(cm.aggregate_measures(per_topic))


def diversity(model, topic_ids):
    """Доля уникальных слов среди топ-слов всех тем."""
    words = [w for t in topic_ids for w in top_words(model, t)]
    return len(set(words)) / len(words)


def compare_with_p5(doc_topics, p5):
    """NMI и ARI между темами BERTopic и topic из P5, без шума (-1).

    Доля шума пишется в метрики отдельно.
    """
    df = doc_topics.join(p5.rename("p5_topic"), on="article_id", how="inner")
    df = df[df["topic"] != -1]
    return {
        "nmi_p5_topic": round(float(normalized_mutual_info_score(df["p5_topic"], df["topic"])), 4),
        "ari_p5_topic": round(float(adjusted_rand_score(df["p5_topic"], df["topic"])), 4),
        "docs_compared": int(len(df)),
    }, df


def dominant_p5_topic(compared):
    """Самая частая тема P5 в каждой теме BERTopic и её доля."""
    counts = compared.groupby(["topic", "p5_topic"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("topic")["n"].transform("sum")
    counts = counts.sort_values("n", ascending=False).drop_duplicates("topic")
    return counts.set_index("topic")[["p5_topic", "share"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    ap.add_argument("--nr-topics", type=int, default=NR_TOPICS)
    args = ap.parse_args()

    out = args.data / "topics"
    reports = args.data / "reports"
    out.mkdir(parents=True, exist_ok=True)
    setup_log(reports)
    log.info(f"=== темы: min_cluster_size={args.min_cluster_size}, nr_topics={args.nr_topics}")
    started = time.time()
    timings = {}

    # данные
    df = load_canonical(args.data / "stories" / "canonical.csv")
    emb = load_embeddings(args.data / "embeddings", df["article_id"])
    lemmas = load_or_build_lemmas(df, out / "lemmas.parquet")
    p5 = load_p5_topics(args.data / "enriched")
    timings["load_s"] = round(time.time() - started)

    # BERTopic
    step = time.time()
    model = build_model(args.min_cluster_size)
    model.fit_transform(lemmas, embeddings=emb)
    n_raw = len([topic for topic in model.topic_sizes_ if topic != -1])
    log.info(f"BERTopic: {n_raw} тем до слияния, {time.time() - step:.0f} с")
    timings["fit_s"] = round(time.time() - step)

    # иерархия и слияние до nr_topics
    step = time.time()
    hierarchy = model.hierarchical_topics(lemmas)
    hierarchy.to_csv(out / "hierarchy.csv", index=False)
    if n_raw > args.nr_topics:
        model.reduce_topics(lemmas, nr_topics=args.nr_topics)
    else:
        log.info(f"тем не больше {args.nr_topics}, слияние не нужно")
    topic_ids = sorted(topic for topic in model.topic_sizes_ if topic != -1)
    log.info(f"после слияния: {len(topic_ids)} тем, {time.time() - step:.0f} с")
    timings["reduce_s"] = round(time.time() - step)

    # doc_topics
    doc_topics = pd.DataFrame({
        "article_id": df["article_id"],
        "topic": model.topics_,
        "probability": np.round(model.probabilities_, 4),
    })
    doc_topics.to_csv(out / "doc_topics.csv", index=False)
    outlier_share = float((doc_topics["topic"] == -1).mean())
    log.info(f"шум HDBSCAN (topic = -1): {outlier_share:.1%} документов")

    # метрики
    step = time.time()
    per_topic_cv, mean_cv = coherence_cv(model, topic_ids, lemmas)
    div = diversity(model, topic_ids)
    p5_scores, compared = compare_with_p5(doc_topics, p5)
    log.info(f"coherence c_v = {mean_cv:.4f}, diversity = {div:.4f}, "
             f"NMI = {p5_scores['nmi_p5_topic']}, ARI = {p5_scores['ari_p5_topic']} "
             f"({p5_scores['docs_compared']} документов), {time.time() - step:.0f} с")
    timings["metrics_s"] = round(time.time() - step)

    # список тем, -1 тоже включаем
    dominant = dominant_p5_topic(compared)
    coherence_of = dict(zip(topic_ids, per_topic_cv))
    topics = pd.DataFrame({
        "ID": list(model.topic_sizes_.keys()),
        "Count": list(model.topic_sizes_.values()),
    }).sort_values("ID")
    topics["Words"] = [", ".join(top_words(model, topic)) for topic in topics["ID"]]
    topics["coherence_cv"] = topics["ID"].map(coherence_of).round(4)
    topics["p5_topic"] = topics["ID"].map(dominant["p5_topic"])
    topics["p5_share"] = topics["ID"].map(dominant["share"]).round(3)
    topics.to_csv(out / "topics.csv", index=False)
    for row in topics.sort_values("Count", ascending=False).head(10).itertuples():
        log.info(f"  тема {row.ID:>4} | {row.Count:>6} | {row.Words}")

    # topics_over_time по неделям (понедельник)
    step = time.time()
    weeks = df["dt_utc"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    # Timestamp, а не строки: BERTopic 0.17 для строк вызывает pd.to_datetime(infer_datetime_format=...), в pandas 3 его нет
    tot = model.topics_over_time(lemmas, weeks.tolist())
    tot.to_csv(out / "topics_over_time.csv", index=False)
    model.visualize_topics_over_time(tot, top_n_topics=TOT_TOP_TOPICS).write_html(
        out / "topics_over_time.html")
    log.info(f"topics_over_time: {tot['Timestamp'].nunique()} недель, {time.time() - step:.0f} с")
    timings["over_time_s"] = round(time.time() - step)

    metrics = {
        "documents": len(df),
        "min_cluster_size": args.min_cluster_size,
        "nr_topics_target": args.nr_topics,
        "umap": {"n_neighbors": UMAP_NEIGHBORS, "n_components": UMAP_COMPONENTS, "seed": SEED},
        "topics_before_merge": n_raw,
        "topics": len(topic_ids),
        "outlier_share": round(outlier_share, 4),
        "coherence_cv": round(mean_cv, 4),
        "diversity": round(div, 4),
        "top_words": TOP_WORDS,
        **p5_scores,
        "timings": timings,
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    }
    (reports / "topics_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"готово за {metrics['elapsed_minutes']} мин, результаты в {out}")
if __name__ == "__main__":
    main()
