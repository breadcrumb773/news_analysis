"""Общие пути, константы и логгер постобработки."""

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("postprocess")

# шаг 2
# порог доли второго типа для type_ambiguous
TYPE_AMBIG_SHARE = 0.2

# шаг 3
# Эти слова не считаются географическим уточнением ORG ("МЧС России" = "МЧС").
GEO_NEUTRAL = frozenset({"россия", "рф", "российский", "федерация", "страна"})
# слова, по которым хвост после "по" считается территорией ("ГУ МЧС по Свердловской области")
REGION_WORDS = frozenset({"область", "край", "округ", "район", "регион", "республика",
                          "город", "столица", "субъект"})
# минимум сюжетов, чтобы однословная LOC/GPE попала в список мест. Ниже много улиц и прилагательных
MIN_PLACE_STORIES = 100

EMB_NEIGHBOURS = 10  # соседей на форму в канале A
EMB_MIN = 0.85  # минимальный косинус
EMB_STEP = 0.02  # шаг повышения порога при разрезании больших компонент
EMB_CAP = 0.99  # максимальный порог
MAX_COMPONENT = 20  # макс. размер компоненты
BLOCK = 2048  # размер блока при умножении матриц в kNN

# шаг 4
BATCH_SIZE = 100  # форм в одном запросе к LLM

ID_PREFIX = {"ORG": "org", "PER": "per", "LOC": "loc", "GPE": "gpe",
             "PRODUCT": "prod", "EVENT": "event"}


class Paths:
    """Пути входов и выходов относительно data/."""

    def __init__(self, data):
        self.data = Path(data)
        self.enriched = self.data / "enriched"  # шарды P5
        self.canonical = self.data / "stories" / "canonical.csv"  # тексты из P4
        self.out = self.data / "entities"
        self.linked = self.data / "enriched_linked"  # шарды P5 с entity_id
        self.reports = self.data / "reports"


def setup_log(reports):
    """Лог в консоль и в data/reports/postprocess_log.txt (дописывается)."""
    reports.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    file = logging.FileHandler(reports / "postprocess_log.txt", encoding="utf-8")
    file.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.handlers = [file, console]
    log.setLevel(logging.INFO)
    return log


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
