"""Шаг 2: чистка поверхностных форм и словарь нормализованных форм forms_norm."""

import re
from functools import lru_cache

import pandas as pd
import pymorphy3

from postprocess.common import TYPE_AMBIG_SHARE, log
from postprocess.load import llm_norm_mode, sample_titles

QUOTES = re.compile(r"[«»\"'„“”‘’`]")
EDGE_PUNCT = " .,;:!?()[]{}-—–…/\\|*#№"
CYRILLIC = re.compile(r"[а-яё]")
LATIN = re.compile(r"[a-z]")

# Аббревиатуры не лемматизируем, pymorphy их портит ("СО ЕЭС" -> "с еэс").
# Аббревиатура = слово в верхнем регистре длиной до 5 букв, длиннее - это капс в заголовке.
# Для PER не применяется, фамилии в заголовках часто капсом.
ABBR = re.compile(r"[A-ZА-ЯЁ][A-ZА-ЯЁ0-9-]{1,4}")
ABBR_TYPES = frozenset({"ORG", "LOC", "GPE", "PRODUCT", "EVENT"})

# мусор из P5: проценты, возраст, числа. В кандидаты не идут
JUNK = re.compile(r"\d+([.,]\d+)?\s*%|\d+-летн|[\d\s.,]+$")

# ОПФ, снимаются только у ORG
LEGAL_FORMS = re.compile(
    r"\b(пао|оао|зао|ао|ооо|нко|ано|фгуп|фгбу|гуп|муп|нао|тоо|ип|"
    r"llc|ltd|inc|plc|corp|gmbh|ag|sa)\b\.?")

morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=None)
def lemma(word):
    """Нормальная форма слова. Слова без кириллицы не трогаем."""
    if not CYRILLIC.search(word):
        return word
    return morph.parse(word)[0].normal_form.replace("ё", "е")


def tokens(name):
    """Разбивка на слова без кавычек, пунктуация снимается с краёв каждого слова.

    Если снимать только с краёв строки, скобка остаётся на слове и ломает лемму.
    """
    s = QUOTES.sub(" ", name.replace(" ", " "))
    return [word for word in (w.strip(EDGE_PUNCT) for w in s.split()) if word]


def simple_form(name):
    """Чистка без лемматизации: кавычки, пунктуация, регистр, ё."""
    return " ".join(tokens(name)).lower().replace("ё", "е")


def keep_as_is(name, type_):
    """Слова, которые не лемматизируются: аббревиатуры и слова со смесью латиницы и кириллицы."""
    keep = set()
    for word in tokens(name):
        low = word.lower().replace("ё", "е")
        if (type_ in ABBR_TYPES and ABBR.fullmatch(word)) or (CYRILLIC.search(low) and LATIN.search(low)):
            keep.add(low)
    return keep


def clean_form(name, type_):
    """Поверхностная форма -> alias_key."""
    keep = keep_as_is(name, type_)
    s = simple_form(name)
    if type_ == "ORG":
        # если после удаления ОПФ ничего не осталось, оставляем как было
        s = " ".join(LEGAL_FORMS.sub(" ", s).split()) or s
    return " ".join(word if word in keep else lemma(word) for word in s.split())


def normalize_forms(forms, mentions, articles):
    """Шаг 2. alias_key для каждой формы и словарь forms_norm.

    Тип нормализованной формы берётся самый частый по упоминаниям. Если второй тип
    набирает долю >= TYPE_AMBIG_SHARE, ставится type_ambiguous.
    """
    forms["alias_key"] = [clean_form(n, t) for n, t in zip(forms["name"], forms["type"])]
    mentions = mentions.merge(forms[["name", "type", "alias_key"]], on=["name", "type"])

    # тип
    by_type = forms.groupby(["alias_key", "type"])["n_mentions"].sum().reset_index()
    by_type["share"] = by_type["n_mentions"] / by_type.groupby("alias_key")["n_mentions"].transform("sum")
    by_type = by_type.sort_values(["alias_key", "share"], ascending=[True, False], kind="stable")
    by_type["rank"] = by_type.groupby("alias_key").cumcount()  # 0 - основной тип
    main_type = by_type[by_type["rank"] == 0].set_index("alias_key")["type"]
    ambiguous = set(by_type.loc[(by_type["rank"] == 1) & (by_type["share"] >= TYPE_AMBIG_SHARE),
                                "alias_key"])

    # словарь
    norm = mentions.groupby("alias_key").agg(
        n_stories=("story_id", "nunique"),
        n_mentions=("article_id", "size"),
    ).reset_index()
    norm["type"] = norm["alias_key"].map(main_type)
    norm["type_ambiguous"] = norm["alias_key"].isin(ambiguous)

    # варианты написания по убыванию частоты, dict.fromkeys убирает повторы (одно имя с разными типами)
    variants = forms.sort_values("n_mentions", ascending=False, kind="stable") \
                    .groupby("alias_key")["name"].agg(lambda s: list(dict.fromkeys(s)))
    norm["surface_variants"] = norm["alias_key"].map(variants)
    norm["top_surface"] = norm["surface_variants"].str[0]

    norm = norm.merge(sample_titles(mentions, ["alias_key"], articles), on="alias_key", how="left")
    norm = norm.merge(llm_norm_mode(mentions, ["alias_key"]), on="alias_key", how="left")

    # в шаги 3-7 идут только активные: больше одного сюжета и не мусор
    junk = [bool(JUNK.match(key)) for key in norm["alias_key"]]
    norm["active"] = (norm["n_stories"] > 1) & ~pd.Series(junk, index=norm.index)

    # form_id = номер строки, сортировка фиксированная, чтобы батчи и кэш LLM совпадали между запусками
    norm = norm.sort_values(["n_stories", "n_mentions", "alias_key"],
                            ascending=[False, False, True]).reset_index(drop=True)
    norm.index.name = "form_id"

    forms["type_ambiguous"] = forms["alias_key"].isin(ambiguous)
    log.info(f"шаг 2: {len(forms)} поверхностных -> {len(norm)} нормализованных форм, "
             f"активных {norm['active'].sum()}, в хвосте {(~norm['active']).sum()}, "
             f"с неоднозначным типом {len(ambiguous)}, отброшено мусорных {sum(junk)}")
    return forms, norm, mentions
