"""Шаг 7: сборка сущностей и итоговые таблицы."""

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from postprocess.common import ID_PREFIX, log
from postprocess.normalize import simple_form


class UnionFind:
    """Union-Find, корнем становится меньший form_id (более частая форма)."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def merge_pairs(accepted):
    """Группы -> все пары form_id внутри группы."""
    pairs = set()
    for group in accepted:
        pairs.update(itertools.combinations(sorted(group["form_ids"]), 2))
    return pd.DataFrame(sorted(pairs), columns=["a", "b"])


def channel_label(channels):
    """Канал пары. transitive ."""
    if not isinstance(channels, list):
        return "transitive"
    return "both" if len(channels) == 2 else channels[0]


def pick_canonical(forms, norm):
    """Каноническое название кластера.

    Приоритет: самое частое написание, у которого simple_form совпадает с alias_key
    (т.е. уже в начальной форме), затем normalized_llm корневой формы, затем top_surface.
    """
    f = forms[["name", "alias_key", "n_mentions", "cluster"]]
    nominative = f[[simple_form(n) == k for n, k in zip(f["name"], f["alias_key"])]]
    first = nominative.sort_values("n_mentions", ascending=False, kind="stable") \
                      .drop_duplicates("cluster").set_index("cluster")["name"]
    roots = norm[norm.index == norm["cluster"]]
    return first.reindex(roots.index).fillna(roots["normalized_llm"]).fillna(roots["top_surface"])


def shares(mentions, column, values):
    """Доли значений column по entity_id, словарь на сущность."""
    table = pd.crosstab(mentions["entity_id"], mentions[column], normalize="index")
    table = table.reindex(columns=values, fill_value=0).round(3)
    return pd.Series(table.to_dict("records"), index=table.index)


def assemble(forms, norm, mentions, accepted, veto, edges):
    """Шаг 7. Возвращает mentions, forms, norm, entities и статистику."""
    # кластеры
    pairs = merge_pairs(accepted)
    uf = UnionFind(len(norm))
    for a, b in zip(pairs["a"], pairs["b"]):
        uf.union(a, b)
    norm["cluster"] = [uf.find(i) for i in range(len(norm))]

    # вклад каналов по принятым парам
    pairs = pairs.merge(edges[["a", "b", "channels"]], on=["a", "b"], how="left")
    contribution = pairs["channels"].map(channel_label).value_counts().to_dict()

    # поля по результатам шагов 3-6
    batch_of, conf_of = {}, {}
    cluster_conf = defaultdict(list)
    for group in accepted:
        for fid in group["form_ids"]:
            batch_of[fid] = group["batch_id"]
            conf_of[fid] = group["confidence"]
        if len(group["form_ids"]) > 1 and group["confidence"] is not None:
            cluster_conf[uf.find(group["form_ids"][0])].append(group["confidence"])

    channels = defaultdict(set)
    for a, b, c in zip(edges["a"], edges["b"], edges["channels"]):
        channels[a].update(c)
        channels[b].update(c)

    norm["batch_id"] = [batch_of.get(i) for i in norm.index]
    norm["llm_confidence"] = [conf_of.get(i) for i in norm.index]
    norm["in_candidates"] = [i in channels for i in norm.index]
    norm["channels"] = [sorted(channels.get(i, ())) for i in norm.index]
    norm["veto_applied"] = norm.index.isin(veto["form_id"].dropna().astype(int))

    # rule - слияние на шаге 2, llm - слияние по ответу LLM, singleton - без слияния
    aliases_in_cluster = norm.groupby("cluster")["alias_key"].transform("size")
    surfaces_in_alias = norm["alias_key"].map(forms.groupby("alias_key").size())
    norm["method"] = np.where(aliases_in_cluster > 1, "llm",
                              np.where(surfaces_in_alias > 1, "rule", "singleton"))

    # entity_id: номер внутри типа по убыванию частоты
    forms = forms.merge(norm[["alias_key", "cluster"]], on="alias_key")
    mentions = mentions.merge(norm[["alias_key", "cluster"]], on="alias_key")
    roots = norm[norm.index == norm["cluster"]]
    clusters = pd.DataFrame({
        "type": roots["type"],
        "canonical": pick_canonical(forms, norm),
        "n_mentions": mentions.groupby("cluster").size().reindex(roots.index, fill_value=0),
    })
    clusters = clusters.sort_values(["type", "n_mentions", "canonical"],
                                    ascending=[True, False, True], kind="stable")
    number = clusters.groupby("type").cumcount() + 1
    clusters["entity_id"] = [f"{ID_PREFIX.get(t, t.lower())}_{k:06d}"
                             for t, k in zip(clusters["type"], number)]
    # минимальный confidence LLM по группам кластера
    clusters["min_confidence"] = [min(cluster_conf[c]) if cluster_conf[c] else None
                                  for c in clusters.index]

    norm["entity_id"] = norm["cluster"].map(clusters["entity_id"])
    norm["canonical"] = norm["cluster"].map(clusters["canonical"])
    link = norm[["alias_key", "entity_id", "canonical", "method"]]

    # упоминания
    mentions = mentions.merge(link, on="alias_key")
    mentions = mentions.merge(forms[["name", "type", "type_ambiguous"]], on=["name", "type"])
    mentions = mentions[[
        "article_id", "story_id", "date", "source",
        "name", "type", "role", "sentiment", "confidence", "grounded",
        "alias_key", "entity_id", "canonical", "normalized_llm", "method", "type_ambiguous",
    ]]

    # формы
    forms = forms.merge(norm[["alias_key", "entity_id", "method", "in_candidates", "channels",
                              "llm_confidence", "veto_applied", "batch_id"]], on="alias_key")
    forms = forms[[
        "name", "type", "n_mentions", "n_docs", "n_stories", "n_sources", "first_seen", "last_seen",
        "conf_mean", "llm_norm_mode", "llm_norm_agree", "type_ambiguous",
        "alias_key", "entity_id", "method", "sample_title",
        "in_candidates", "channels", "llm_confidence", "veto_applied", "batch_id",
    ]]

    # сущности
    entities = mentions.groupby("entity_id").agg(
        n_mentions=("article_id", "size"),
        n_docs=("article_id", "nunique"),
        n_stories=("story_id", "nunique"),
        n_sources=("source", "nunique"),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
    )
    by_id = clusters.set_index("entity_id")
    entities.insert(0, "type", by_id["type"])
    entities.insert(0, "canonical", by_id["canonical"])
    aliases = forms.sort_values("n_mentions", ascending=False, kind="stable") \
                   .groupby("entity_id")["name"].agg(lambda s: list(dict.fromkeys(s)))
    entities.insert(2, "aliases", aliases)
    entities.insert(3, "n_aliases", entities["aliases"].str.len())
    entities["sentiment_dist"] = shares(mentions, "sentiment", ["pos", "neu", "neg", "unknown"])
    entities["top_roles"] = shares(mentions, "role", ["actor", "object", "mentioned"])
    entities["ambiguous"] = norm.groupby("entity_id")["type_ambiguous"].any()
    methods = pd.crosstab(forms["entity_id"], forms["method"]) \
                .reindex(columns=["rule", "llm", "singleton"], fill_value=0)
    entities["methods"] = pd.Series(methods.to_dict("records"), index=methods.index)
    entities["min_confidence"] = by_id["min_confidence"]
    entities["n_veto_splits"] = norm.groupby("entity_id")["veto_applied"].sum().astype(int)
    entities = entities.sort_values("n_mentions", ascending=False).reset_index()

    stats = {
        "surface_forms": len(forms),
        "normalized_forms": len(norm),
        "entities": len(entities),
        "merge_pairs": len(pairs),
        "channel_contribution": contribution,
    }
    log.info(f"шаг 7: {stats['surface_forms']} форм -> {stats['normalized_forms']} нормализованных "
             f"-> {stats['entities']} сущностей; вклад каналов в слияния: {contribution}")
    return mentions, forms, norm, entities, stats
