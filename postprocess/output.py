"""Сохранение таблиц, шарды с entity_id и отчёты."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from enrich import ENTITY_STRUCT, SHARD_SCHEMA
from postprocess.common import log, write_json

LINK_FIELDS = ["alias_key", "entity_id", "canonical", "method"]


def save_tables(paths, mentions, forms, norm, entities, veto):
    """Сохраняет таблицы в data/entities/."""
    paths.out.mkdir(parents=True, exist_ok=True)
    mentions.to_parquet(paths.out / "mentions.parquet", index=False)
    forms.to_parquet(paths.out / "forms.parquet", index=False)
    norm.reset_index().to_parquet(paths.out / "forms_norm.parquet", index=False)
    entities.to_parquet(paths.out / "entities.parquet", index=False)
    veto.to_csv(paths.reports / "postprocess_veto_log.csv", index=False)
    log.info(f"таблицы: {paths.out} (mentions, forms, forms_norm, entities)")


def link_enriched(paths, forms, norm):
    """Копии шардов P5 в data/enriched_linked/, у каждой сущности добавлены LINK_FIELDS."""
    entity = pa.struct(list(ENTITY_STRUCT) + [pa.field(name, pa.string()) for name in LINK_FIELDS])
    schema = pa.schema([pa.field("entities", pa.list_(entity)) if f.name == "entities" else f
                        for f in SHARD_SCHEMA])
    link = forms[["name", "type", "alias_key", "entity_id", "method"]] \
        .merge(norm[["alias_key", "canonical"]], on="alias_key")
    lookup = {(name, type_): dict(zip(LINK_FIELDS, values))
              for name, type_, *values in link[["name", "type"] + LINK_FIELDS].itertuples(index=False)}
    empty = dict.fromkeys(LINK_FIELDS)

    paths.linked.mkdir(parents=True, exist_ok=True)
    for path in tqdm(sorted(paths.enriched.glob("shard_*.parquet")), desc="шарды", unit="шард"):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["entities"] = [{**e, **lookup.get((e["name"], e["type"]), empty)}
                               for e in row["entities"] or []]
        tmp = paths.linked / (path.name + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema), tmp, compression="zstd")
        tmp.replace(paths.linked / path.name)
    log.info(f"дополненное обогащение: {paths.linked}")


def write_reports(paths, forms, norm, entities, stats):
    """Отчёты в data/reports: summary, сжатие словаря по типам, топ-20 сущностей."""
    # сжатие по типам; у norm и entities тип основной, поэтому суммы по столбцам могут не сходиться
    compression = pd.DataFrame({
        "surface_forms": forms.groupby("type").size(),
        "normalized_forms": norm.groupby("type").size(),
        "entities": entities.groupby("type").size(),
    }).fillna(0).astype(int)
    compression.loc["ВСЕГО"] = [len(forms), len(norm), len(entities)]
    compression["surface_to_entities"] = (compression["surface_forms"]
                                          / compression["entities"].clip(lower=1)).round(2)
    compression.index.name = "type"
    compression.to_csv(paths.reports / "postprocess_compression.csv")

    # топ-20 сущностей со всеми вариантами написания
    top = entities.head(20)[["entity_id", "canonical", "type", "n_mentions", "n_stories",
                             "n_aliases", "aliases", "methods"]].copy()
    top["aliases"] = top["aliases"].map(" | ".join)
    top.to_csv(paths.reports / "postprocess_top20.csv", index=False)

    summary = {
        **stats,
        "active_forms": int(norm["active"].sum()),
        "tail_forms": int((~norm["active"]).sum()),
        "type_ambiguous_forms": int(norm["type_ambiguous"].sum()),
        "compression_surface_to_normalized": round(len(forms) / max(len(norm), 1), 3),
        "compression_normalized_to_entities": round(len(norm) / max(len(entities), 1), 3),
        "entities_merged_by_llm": int((entities["methods"].str["llm"] > 0).sum()),
    }
    write_json(paths.reports / "postprocess_summary.json", summary)
    log.info(f"отчёты: postprocess_summary.json, postprocess_compression.csv, postprocess_top20.csv, postprocess_veto_log.csv "
             f"в {paths.reports}")
    log.info("\n" + compression.to_string())
