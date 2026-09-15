"""Граф сущностей: центральности, сообщества, изменения по месяцам.

Узлы - сущности P6 (entity_id). Ребро - совместное появление в одном сюжете (не статье).
Сущности сюжета = сущности его канонического документа.

Вес ребра PMI(a, b) = log2(n_ab * N / (n_a * n_b)), N - число сюжетов, n_a, n_b - сюжетов с сущностью,
n_ab - с обеими. У редких пар PMI завышен, поэтому пороги MIN_PAIR и MIN_STORIES.

Считается: degree, betweenness, PageRank, сообщества Louvain, модулярность, NMI с типами
и темами P5, графы по месяцам (размер, модулярность, новые сильные рёбра), визуализация pyvis и PNG.

Вход: data/entities/mentions.parquet, data/entities/entities.parquet, data/enriched/shard_*.parquet (topic).
Выход в data/graph/: nodes.csv, edges.csv, communities.csv, monthly_stats.csv, monthly_top.csv, new_edges.csv.
В data/reports/: graph_log.txt, graph_summary.json, graph_*.csv, graph_*.png, graph_map.html.

uv run python entity_graph.py
uv run python entity_graph.py --min-stories 20 --min-pair 10
"""

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyvis.network import Network
from sklearn.metrics import normalized_mutual_info_score

DATA = Path(__file__).resolve().parent.parent / "data"

# общий граф
MIN_STORIES = 10  # мин. сюжетов у сущности
MIN_PAIR = 5  # мин. общих сюжетов у пары
MIN_PMI = 1.0

# графы по месяцам
MONTH_MIN_STORIES = 3
MONTH_MIN_PAIR = 3
NEW_EDGE_MIN = 5  # мин. сюжетов за месяц для сильного нового ребра
NEW_EDGES_TOP = 10

BETWEENNESS_EXACT = 3000  # выше - betweenness по выборке из BETWEENNESS_K узлов
BETWEENNESS_K = 1000
SEED = 42
TOP = 20
TOP_MONTH = 10
PLOT_NODES = 150
PLOT_LABELS = 30
HTML_NODES = 400

# палитра как в trends.py, сообщества после 8-го серые
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#898781"
GRID = "#e6e5e1"

log = logging.getLogger("graph")


def setup_log(reports):
    """Лог в консоль и в data/reports/graph_log.txt."""
    reports.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    file = logging.FileHandler(reports / "graph_log.txt", encoding="utf-8")
    file.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.handlers = [file, console]
    log.setLevel(logging.INFO)


def load_story_entities(folder):
    """Уникальные пары (story_id, entity_id) и месяц сюжета (по дате в P6, UTC)."""
    path = folder / "mentions.parquet"
    if not path.exists():
        raise SystemExit(f"нет {path}, сначала запустите postprocess.py (P6)")
    mentions = pd.read_parquet(path, columns=["story_id", "entity_id", "date"])
    log.info(f"упоминаний: {len(mentions)}, сущностей: {mentions['entity_id'].nunique()}, "
             f"сюжетов с сущностями: {mentions['story_id'].nunique()}")

    mentions["month"] = pd.to_datetime(mentions["date"]).dt.to_period("M").astype(str)
    month = mentions.groupby("story_id")["month"].min()
    pairs = mentions[["story_id", "entity_id"]].drop_duplicates()
    pairs["month"] = pairs["story_id"].map(month)
    return pairs


def load_entity_names(folder):
    """entity_id -> canonical, type."""
    entities = pd.read_parquet(folder / "entities.parquet", columns=["entity_id", "canonical", "type"])
    return entities.set_index("entity_id")


def load_story_topics(folder):
    """story_id -> topic из P5 (ok=True)."""
    paths = sorted(folder.glob("shard_*.parquet"))
    if not paths:
        raise SystemExit(f"в {folder} нет шардов shard_*.parquet, сначала запустите enrich.py (P5)")
    df = pd.concat([pd.read_parquet(p, columns=["story_id", "topic", "ok"]) for p in paths],
                   ignore_index=True)
    df = df[df["ok"]].drop_duplicates("story_id")
    return df.set_index("story_id")["topic"]


def entity_main_topic(story_entities, topics):
    """Самая частая тема сюжетов сущности и её доля."""
    part = story_entities[["story_id", "entity_id"]].copy()
    part["topic"] = part["story_id"].map(topics)
    part = part.dropna(subset=["topic"])
    counts = part.groupby(["entity_id", "topic"]).size().rename("n").reset_index()
    total = counts.groupby("entity_id")["n"].transform("sum")
    counts["topic_share"] = (counts["n"] / total).round(3)
    main = counts.sort_values("n", ascending=False, kind="stable").drop_duplicates("entity_id")
    return main.set_index("entity_id")[["topic", "topic_share"]]

def cooccurrence_edges(story_entities, min_stories, min_pair, min_pmi):
    """Рёбра с весом PMI. Возвращает (edges[a, b, n_stories, pmi], частоты сущностей, N сюжетов)."""
    # фильтр редких сущностей
    n_entity = story_entities.groupby("entity_id")["story_id"].nunique()
    frequent = n_entity[n_entity >= min_stories].index
    part = story_entities[story_entities["entity_id"].isin(frequent)]
    n_total = part["story_id"].nunique()

    # частоты пар
    pair_counts = Counter()
    for ids in part.groupby("story_id")["entity_id"].agg(sorted):
        pair_counts.update(combinations(ids, 2))

    edges = pd.DataFrame([(a, b, n) for (a, b), n in pair_counts.items() if n >= min_pair],
                         columns=["a", "b", "n_stories"])
    if edges.empty:
        return edges.assign(pmi=[]), n_entity[frequent], n_total

    # PMI
    n_a = edges["a"].map(n_entity)
    n_b = edges["b"].map(n_entity)
    edges["pmi"] = [round(math.log2(n * n_total / (x * y)), 3)
                    for n, x, y in zip(edges["n_stories"], n_a, n_b)]
    edges = edges[edges["pmi"] >= min_pmi].reset_index(drop=True)
    return edges, n_entity[frequent], n_total

def build_graph(edges, names):
    """Граф networkx. distance = 1 / PMI для betweenness."""
    graph = nx.Graph()
    for a, b, n, pmi in edges[["a", "b", "n_stories", "pmi"]].itertuples(index=False):
        graph.add_edge(a, b, n_stories=int(n), pmi=float(pmi), distance=1.0 / pmi)
    for node in graph.nodes:
        graph.nodes[node]["canonical"] = names["canonical"].get(node, node)
        graph.nodes[node]["type"] = names["type"].get(node, "?")
    return graph

def find_communities(graph):
    """Louvain по PMI, сообщества пронумерованы по убыванию размера."""
    parts = nx.community.louvain_communities(graph, weight="pmi", seed=SEED)
    parts = sorted(parts, key=len, reverse=True)
    membership = {node: k for k, part in enumerate(parts) for node in part}
    modularity = nx.community.modularity(graph, parts, weight="pmi")
    return membership, parts, modularity

def centralities(graph, membership):
    """Таблица узлов: degree, strength (сумма PMI), betweenness, pagerank,
    n_neighbor_communities (число других сообществ среди соседей)."""
    if graph.number_of_nodes() > BETWEENNESS_EXACT:
        log.info(f"betweenness по выборке из {BETWEENNESS_K} опорных узлов "
                 f"(узлов {graph.number_of_nodes()} > {BETWEENNESS_EXACT})")
        betweenness = nx.betweenness_centrality(graph, k=BETWEENNESS_K, weight="distance", seed=SEED)
    else:
        betweenness = nx.betweenness_centrality(graph, weight="distance")
    pagerank = nx.pagerank(graph, weight="pmi")
    degree_c = nx.degree_centrality(graph)

    rows = []
    for node in graph.nodes:
        own = membership[node]
        other = {membership[nb] for nb in graph.neighbors(node)} - {own}
        rows.append({
            "entity_id": node,
            "canonical": graph.nodes[node]["canonical"],
            "type": graph.nodes[node]["type"],
            "degree": graph.degree(node),
            "degree_centrality": round(degree_c[node], 5),
            "strength": round(graph.degree(node, weight="pmi"), 2),
            "betweenness": round(betweenness[node], 6),
            "pagerank": round(pagerank[node], 6),
            "community": own,
            "n_neighbor_communities": len(other),
        })
    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)


def describe_communities(nodes):
    """Сводка по сообществам: размер, топ сущностей, типы, main_topic."""
    rows = []
    for k, part in nodes.groupby("community"):
        part = part.sort_values("pagerank", ascending=False)
        types = part["type"].value_counts(normalize=True).head(3)
        topics = part["topic"].value_counts(normalize=True)
        rows.append({
            "community": k,
            "size": len(part),
            "top_entities": ", ".join(part["canonical"].head(10)),
            "types": ", ".join(f"{t} {s:.0%}" for t, s in types.items()),
            "main_topic": topics.index[0] if len(topics) else None,
            "main_topic_share": round(float(topics.iloc[0]), 3) if len(topics) else None,
            "pagerank_sum": round(float(part["pagerank"].sum()), 4),
        })
    return pd.DataFrame(rows)


def nmi_scores(nodes):
    """NMI сообществ с типами сущностей и с темами."""
    with_topic = nodes.dropna(subset=["topic"])
    return {
        "nmi_type": round(float(normalized_mutual_info_score(nodes["type"], nodes["community"])), 4),
        "nmi_topic": round(float(normalized_mutual_info_score(with_topic["topic"], with_topic["community"])), 4),
        "nodes_with_topic": len(with_topic),
    }


def monthly_evolution(story_entities, names):
    """Графы по месяцам: размер, модулярность, топ PageRank, новые рёбра.

    Новое ребро - не было ни в одном из предыдущих месяцев (для первого месяца не считается).
    Сильное - n_stories >= NEW_EDGE_MIN.
    """
    stats, tops, new_edges = [], [], []
    seen = set()  # рёбра предыдущих месяцев

    for i, month in enumerate(sorted(story_entities["month"].dropna().unique())):
        part = story_entities[story_entities["month"] == month]
        edges, _, n_total = cooccurrence_edges(part, MONTH_MIN_STORIES, MONTH_MIN_PAIR, MIN_PMI)
        row = {"month": month, "stories": part["story_id"].nunique(), "nodes": 0, "edges": 0,
               "density": 0.0, "components": 0, "largest_component_share": 0.0,
               "communities": 0, "modularity": None, "new_edges": 0, "new_strong_edges": 0}
        if edges.empty:
            stats.append(row)
            continue

        graph = build_graph(edges, names)
        membership, parts, modularity = find_communities(graph)
        largest = max(nx.connected_components(graph), key=len)
        row.update(nodes=graph.number_of_nodes(), edges=graph.number_of_edges(),
                   density=round(nx.density(graph), 5),
                   components=nx.number_connected_components(graph),
                   largest_component_share=round(len(largest) / graph.number_of_nodes(), 3),
                   communities=len(parts), modularity=round(modularity, 4))

        pagerank = pd.Series(nx.pagerank(graph, weight="pmi")).nlargest(TOP_MONTH)
        for rank, (node, value) in enumerate(pagerank.items(), start=1):
            tops.append({"month": month, "rank": rank, "entity_id": node,
                         "canonical": graph.nodes[node]["canonical"], "pagerank": round(value, 5)})

        keys = list(zip(edges["a"], edges["b"]))
        if i > 0:
            fresh = edges[[key not in seen for key in keys]]
            strong = fresh[fresh["n_stories"] >= NEW_EDGE_MIN]
            row.update(new_edges=len(fresh), new_strong_edges=len(strong))
            strong = strong.sort_values(["n_stories", "pmi"], ascending=False).head(NEW_EDGES_TOP)
            for a, b, n, pmi in strong[["a", "b", "n_stories", "pmi"]].itertuples(index=False):
                new_edges.append({"month": month, "a": a, "b": b,
                                  "canonical_a": graph.nodes[a]["canonical"],
                                  "canonical_b": graph.nodes[b]["canonical"],
                                  "n_stories": n, "pmi": pmi})
        seen.update(keys)
        stats.append(row)
        log.info(f"  {month}: сюжетов {row['stories']}, узлов {row['nodes']}, рёбер {row['edges']}, "
                 f"сообществ {row['communities']}, модулярность {row['modularity']}, "
                 f"новых сильных рёбер {row['new_strong_edges']}")

    return pd.DataFrame(stats), pd.DataFrame(tops), pd.DataFrame(new_edges)


def community_color(k):
    return COLORS[k] if k < len(COLORS) else GRAY


def plot_graph(graph, nodes, path):
    """PNG с PLOT_NODES узлами по PageRank, цвет - сообщество, размер - PageRank."""
    top = nodes.head(PLOT_NODES).set_index("entity_id")
    sub = graph.subgraph(top.index)
    pos = nx.spring_layout(sub, weight="pmi", seed=SEED, k=1.5 / math.sqrt(max(len(sub), 1)))
    sizes = top["pagerank"] / top["pagerank"].max() * 600 + 20

    fig, ax = plt.subplots(figsize=(14, 11))
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=GRID, width=0.6)
    nx.draw_networkx_nodes(sub, pos, ax=ax, nodelist=list(top.index),
                           node_size=list(sizes),
                           node_color=[community_color(k) for k in top["community"]],
                           edgecolors="white", linewidths=0.8)
    labels = {e: top.loc[e, "canonical"] for e in top.index[:PLOT_LABELS]}
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=9)
    ax.set_title(f"Граф сущностей: топ-{len(sub)} узлов по PageRank, цвет - сообщество")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_monthly(stats, path):
    """Размер графа и модулярность по месяцам."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(stats["month"], stats["nodes"], color=COLORS[0], linewidth=2, marker="o", label="узлы")
    ax1.plot(stats["month"], stats["edges"], color=COLORS[1], linewidth=2, marker="o", label="рёбра")
    ax1.set_title("Размер графа по месяцам")
    ax1.legend(frameon=False)
    ax2.plot(stats["month"], stats["modularity"], color=COLORS[2], linewidth=2, marker="o")
    ax2.set_title("Модулярность разбиения Louvain")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_html(graph, nodes, path):
    """HTML через pyvis для HTML_NODES узлов.

    Пишем через write_text(utf-8), pyvis write_html использует системную кодировку.
    """
    top = nodes.head(HTML_NODES).set_index("entity_id")
    net = Network(height="850px", width="100%", bgcolor="#ffffff", cdn_resources="remote")
    max_pr = top["pagerank"].max()
    for node, r in top.iterrows():
        net.add_node(node, label=r["canonical"], color=community_color(r["community"]),
                     size=float(8 + 40 * r["pagerank"] / max_pr),
                     title=(f"{r['canonical']} ({r['type']})\nсообщество {r['community']}\n"
                            f"тема: {r['topic']}\nстепень {r['degree']}, "
                            f"PageRank {r['pagerank']:.4f}, betweenness {r['betweenness']:.4f}"))
    for a, b, data in graph.subgraph(top.index).edges(data=True):
        net.add_edge(a, b, value=data["pmi"], color=GRID,
                     title=f"{data['n_stories']} сюжетов, PMI {data['pmi']:.2f}")
    net.barnes_hut(gravity=-8000, spring_length=120)
    path.write_text(net.generate_html(), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Граф сущностей")
    ap.add_argument("--data", default=DATA, help="корень data/")
    ap.add_argument("--min-stories", type=int, default=MIN_STORIES, help="мин. сюжетов у сущности")
    ap.add_argument("--min-pair", type=int, default=MIN_PAIR, help="мин. общих сюжетов у пары")
    ap.add_argument("--min-pmi", type=float, default=MIN_PMI, help="мин. PMI ребра (log2)")
    args = ap.parse_args()

    data = Path(args.data)
    out = data / "graph"
    reports = data / "reports"
    setup_log(reports)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log.info("=" * 70)
    log.info(f"старт: data={data}, min_stories={args.min_stories}, "
             f"min_pair={args.min_pair}, min_pmi={args.min_pmi}")

    story_entities = load_story_entities(data / "entities")
    names = load_entity_names(data / "entities")
    topics = load_story_topics(data / "enriched")
    main_topic = entity_main_topic(story_entities, topics)

    # общий граф
    edges, n_entity, n_total = cooccurrence_edges(story_entities, args.min_stories,
                                                  args.min_pair, args.min_pmi)
    if edges.empty:
        raise SystemExit("после фильтрации нет рёбер, уменьшите --min-pair / --min-pmi")
    graph = build_graph(edges, names)
    largest = max(nx.connected_components(graph), key=len)
    log.info(f"сущностей с порогом {args.min_stories} сюжетов: {len(n_entity)}; "
             f"сюжетов с ними: {n_total}")
    log.info(f"граф: узлов {graph.number_of_nodes()}, рёбер {graph.number_of_edges()}, "
             f"плотность {nx.density(graph):.5f}, компонент {nx.number_connected_components(graph)}, "
             f"крупнейшая {len(largest)} узлов")

    membership, parts, modularity = find_communities(graph)
    nodes = centralities(graph, membership)
    nodes["n_stories"] = nodes["entity_id"].map(n_entity)
    nodes = nodes.join(main_topic, on="entity_id")
    nmi = nmi_scores(nodes)
    log.info(f"Louvain: сообществ {len(parts)}, модулярность {modularity:.4f}; "
             f"NMI с типами {nmi['nmi_type']}, NMI с темами {nmi['nmi_topic']}")

    communities = describe_communities(nodes)
    edges["canonical_a"] = edges["a"].map(names["canonical"])
    edges["canonical_b"] = edges["b"].map(names["canonical"])
    nodes.to_csv(out / "nodes.csv", index=False)
    edges.sort_values("pmi", ascending=False).to_csv(out / "edges.csv", index=False)
    communities.to_csv(out / "communities.csv", index=False)

    # топы по центральностям
    report_cols = ["entity_id", "canonical", "type", "n_stories", "degree", "betweenness",
                   "pagerank", "community", "n_neighbor_communities", "topic"]
    for column in ["degree", "betweenness", "pagerank"]:
        top = nodes.sort_values(column, ascending=False).head(TOP)[report_cols]
        top.to_csv(reports / f"graph_top_{column}.csv", index=False)
        log.info(f"топ-10 по {column}: " + ", ".join(top["canonical"].head(10)))
    communities.head(TOP).to_csv(reports / "graph_communities.csv", index=False)
    log.info("\n" + communities.head(10)[["community", "size", "main_topic", "main_topic_share",
                                          "top_entities"]].to_string(index=False))

    # топ рёбер по PMI среди 10% самых частых пар
    strong = edges[edges["n_stories"] >= edges["n_stories"].quantile(0.9)]
    strong.sort_values("pmi", ascending=False).head(TOP).to_csv(reports / "graph_top_edges.csv", index=False)

    log.info("графы по месяцам:")
    monthly, monthly_top, new_edges = monthly_evolution(story_entities, names)
    monthly.to_csv(out / "monthly_stats.csv", index=False)
    monthly_top.to_csv(out / "monthly_top.csv", index=False)
    new_edges.to_csv(out / "new_edges.csv", index=False)

    plot_graph(graph, nodes, reports / "graph_map.png")
    plot_monthly(monthly, reports / "graph_monthly.png")
    save_html(graph, nodes, reports / "graph_map.html")

    summary = {
        "params": {"min_stories": args.min_stories, "min_pair": args.min_pair, "min_pmi": args.min_pmi,
                   "month_min_stories": MONTH_MIN_STORIES, "month_min_pair": MONTH_MIN_PAIR,
                   "new_edge_min": NEW_EDGE_MIN},
        "entities_total": int(story_entities["entity_id"].nunique()),
        "entities_frequent": len(n_entity),
        "stories": n_total,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 6),
        "components": nx.number_connected_components(graph),
        "largest_component_share": round(len(largest) / graph.number_of_nodes(), 4),
        "communities": len(parts),
        "communities_size_5plus": int((communities["size"] >= 5).sum()),
        "modularity": round(modularity, 4),
        **nmi,
        "top_pagerank": list(nodes["canonical"].head(10)),
        "top_betweenness": list(nodes.sort_values("betweenness", ascending=False)["canonical"].head(10)),
        "months": len(monthly),
        # NaN -> null
        "modularity_by_month": {m: (None if pd.isna(q) else float(q))
                                for m, q in zip(monthly["month"], monthly["modularity"])},
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    (reports / "graph_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    log.info(f"таблицы: {out}; отчёты и графики: {reports}/graph_*")
    log.info(f"готово за {summary['elapsed_minutes']} мин")


if __name__ == "__main__":
    main()
