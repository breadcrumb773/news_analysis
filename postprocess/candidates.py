"""Шаг 3: кандидаты на слияние (каналы A и C) и группы кандидатов."""

import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from embed import embed, normalize
from postprocess.common import (BLOCK, EMB_CAP, EMB_MIN, EMB_NEIGHBOURS, EMB_STEP, GEO_NEUTRAL,
                                MAX_COMPONENT, MIN_PLACE_STORIES, REGION_WORDS, log)
from postprocess.normalize import clean_form

EDGE_COLUMNS = ["a", "b", "channel", "score"]


def form_vectors(keys, out, api, model, batch=256):
    """Эмбеддинги alias_key через vLLM (USER-bge-m3). Кэш на диске, сбрасывается при смене списка ключей."""
    vec_path = out / "form_vectors.npy"
    keys_path = out / "form_vectors_keys.txt"
    if vec_path.exists() and keys_path.exists():
        if keys_path.read_text(encoding="utf-8").split("\n") == keys:
            log.info("канал A: векторы форм взяты из кэша")
            return np.load(vec_path)

    log.info(f"канал A: кодируем {len(keys)} форм через {api}")
    parts = []
    try:
        for i in tqdm(range(0, len(keys), batch), desc="канал A", unit="пачка"):
            parts.append(normalize(embed(keys[i:i + batch], api, model)))
    except Exception as exc:
        raise SystemExit(f"эмбеддер не отвечает: {exc}\n"
                         f"нужен vLLM с USER-bge-m3 (docs/run_embeddings.md), "
                         f"либо запуск с --no-emb")
    vectors = np.concatenate(parts)
    out.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    keys_path.write_text("\n".join(keys), encoding="utf-8")
    return vectors


def channel_emb(active, vectors):
    """Канал A: до EMB_NEIGHBOURS ближайших соседей того же типа с косинусом >= EMB_MIN.

    active - активные формы, колонка pos - номер строки в vectors.
    """
    a, b, s = [], [], []
    for _, part in active.groupby("type"):
        ids = part.index.to_numpy()  # form_id
        m = vectors[part["pos"].to_numpy()]
        k = min(EMB_NEIGHBOURS, len(ids) - 1)
        if k < 1:
            continue
        for start in range(0, len(ids), BLOCK):
            sim = m[start:start + BLOCK] @ m.T
            rows = np.arange(len(sim))
            sim[rows, start + rows] = -1.0  # исключаем саму форму
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            score = sim[rows[:, None], top]
            keep = score >= EMB_MIN
            src = ids[start + np.nonzero(keep)[0]]
            dst = ids[top[keep]]
            a.append(np.minimum(src, dst))
            b.append(np.maximum(src, dst))
            s.append(score[keep])
    if not a:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    edges = pd.DataFrame({"a": np.concatenate(a), "b": np.concatenate(b),
                          "channel": "emb", "score": np.concatenate(s).round(4)})
    # пары дублируются (a-b и b-a)
    return edges.drop_duplicates(["a", "b"])[EDGE_COLUMNS]


def gazetteer(norm):
    """Однословные LOC/GPE из данных, используются как список мест."""
    place = norm[norm["type"].isin(("LOC", "GPE")) & norm["active"]
                 & (norm["n_stories"] >= MIN_PLACE_STORIES)]
    return {key for key in place["alias_key"] if " " not in key and len(key) >= 3} - GEO_NEUTRAL


def geo_marker(alias_key, places):
    """Гео-уточнение в названии: слова после "по" и названия мест.

    "гу мчс по москва" -> {москва}, "мчс россия" -> пусто.
    """
    words = alias_key.split()
    tail = set(words[words.index("по") + 1:]) if "по" in words else set()
    if not (tail & places or tail & REGION_WORDS):
        tail = set()  # "по чрезвычайный ситуация" - не территория
    return frozenset((tail | {w for w in words if w in places}) - GEO_NEUTRAL)


def drop_geo_conflicts(edges, active, norm):
    """Удаляет рёбра между ORG с разными гео-уточнениями ("мчс" и "гу мчс по москва").

    По косинусу такие пары близки, и LLM их сливает. Фильтр применяется к обоим каналам,
    в P5 нормализация тоже часто сводит региональное ведомство к федеральному.
    """
    if edges.empty:
        return edges
    places = gazetteer(norm)
    org = active[active["type"] == "ORG"]
    marker = {fid: geo_marker(key, places) for fid, key in zip(org.index, org["alias_key"])}
    keep = [a not in marker or b not in marker or marker[a] == marker[b]
            for a, b in zip(edges["a"], edges["b"])]
    dropped = len(keep) - sum(keep)
    if dropped:
        log.info(f"география: снято {dropped} рёбер между ORG с разными уточнениями")
    return edges[keep].reset_index(drop=True)


def channel_llm(active):
    """Канал C: формы одного типа с одинаковой нормализацией из P5.

    normalized_llm чистится через clean_form и сравнивается как с normalized_llm других
    форм, так и с их alias_key.
    """
    votes = defaultdict(set)
    for fid, key, type_, llm in zip(active.index, active["alias_key"], active["type"],
                                    active["normalized_llm"]):
        votes[(type_, key)].add(fid)
        if isinstance(llm, str) and llm.strip():
            votes[(type_, clean_form(llm, type_))].add(fid)

    a, b = [], []
    for members in votes.values():
        members = sorted(members)
        if len(members) <= MAX_COMPONENT:
            pairs = itertools.combinations(members, 2)
        else:
            # для большой группы звезда вместо полного графа
            pairs = ((members[0], other) for other in members[1:])
        for x, y in pairs:
            a.append(x)
            b.append(y)
    return pd.DataFrame({"a": a, "b": b, "channel": "llm_vote", "score": np.nan})[EDGE_COLUMNS]


def merge_edges(emb, llm):
    """Объединение рёбер каналов: пара, список каналов, косинус."""
    edges = pd.concat([emb, llm], ignore_index=True)
    edges = edges.groupby(["a", "b"]).agg(
        channels=("channel", lambda c: sorted(set(c))),
        score=("score", "max"),
    ).reset_index()
    return edges


def split(sub, thr):
    """Рекурсивно режет компоненту, повышая порог косинуса, пока размер > MAX_COMPONENT.

    Рёбра llm_vote не удаляются.
    """
    if sub.number_of_nodes() <= MAX_COMPONENT or thr >= EMB_CAP:
        return [set(sub)]
    thr = round(thr + EMB_STEP, 4)
    cut = sub.copy()
    cut.remove_edges_from([(u, v) for u, v, d in cut.edges(data=True)
                           if "llm_vote" not in d["channels"] and d["score"] < thr])
    return [part for group in nx.connected_components(cut)
            for part in split(cut.subgraph(group), thr)]


def candidate_groups(edges):
    """Компоненты связности графа -> группы размером не больше MAX_COMPONENT."""
    graph = nx.Graph()
    graph.add_edges_from((a, b, {"channels": c, "score": s})
                         for a, b, c, s in zip(edges["a"], edges["b"], edges["channels"], edges["score"]))

    raw = [sorted(c) for c in nx.connected_components(graph)]
    oversized = [c for c in raw if len(c) > MAX_COMPONENT]
    groups = [c for c in raw if len(c) <= MAX_COMPONENT]
    for comp in tqdm(oversized, desc="хайрбол", disable=not oversized):
        for part in split(graph.subgraph(comp), EMB_MIN):
            part = sorted(part)
            # остаток (связан рёбрами канала C) режем на куски подряд
            groups.extend(part[i:i + MAX_COMPONENT] for i in range(0, len(part), MAX_COMPONENT))
    groups = sorted((g for g in groups if len(g) > 1), key=lambda g: g[0])

    stats = {
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "components": len(raw),
        "oversized_components": len(oversized),
        "largest_component_before": max(map(len, raw), default=0),
        "largest_group_after": max(map(len, groups), default=0),
        "groups": len(groups),
        "forms_in_groups": sum(map(len, groups)),
    }
    return groups, stats


def find_candidates(norm, paths, args):
    """Шаг 3. Возвращает группы (списки form_id), рёбра и статистику."""
    active = norm[norm["active"]].copy()
    active["pos"] = np.arange(len(active))

    if args.no_emb:
        log.info("канал A пропущен (--no-emb)")
        emb = pd.DataFrame(columns=EDGE_COLUMNS)
    else:
        vectors = form_vectors(active["alias_key"].tolist(), paths.out, args.emb_api, args.emb_model)
        emb = channel_emb(active, vectors)
    llm = channel_llm(active)
    edges = merge_edges(emb, llm)
    before = len(edges)
    edges = drop_geo_conflicts(edges, active, norm)

    groups, stats = candidate_groups(edges)
    stats["edges_emb"] = len(emb)
    stats["edges_llm_vote"] = len(llm)
    stats["edges_geo_dropped"] = before - len(edges)
    log.info(f"шаг 3: рёбер A {len(emb)}, C {len(llm)}, после слияния {len(edges)}; "
             f"компонент {stats['components']}, разрезано крупных {stats['oversized_components']} "
             f"({stats['largest_component_before']} -> {stats['largest_group_after']} форм); "
             f"групп в модель {stats['groups']}, форм в них {stats['forms_in_groups']}")
    return groups, edges, stats
