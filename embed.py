"""Эмбеддинги статей через vLLM (USER-bge-m3).

Вход:  data/clean/news_2016.csv      (source, dt_utc, title, text)
Выход: data/embeddings/matrix.npy    (float32, строка = статья)
       data/embeddings/id_map.csv    (row, article_id, source, dt_utc)

Перед запуском в поде должен работать vllm (см. docs/run_embeddings.md):
    vllm serve deepvk/USER-bge-m3 --served-model-name USER-bge-m3 --runner pooling \
        --max-model-len 8192 --gpu-memory-utilization 0.30 --host 127.0.0.1 --port 8000
 
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

DATA = Path(__file__).resolve().parent.parent / "data"
TEXT_LIMIT = 1500   # сколько знаков текста берём после заголовка
TIMEOUT = 600       # таймаут одного запроса, секунды
RETRIES = 5         # попыток на пачку, пауза между ними удваивается


def build_inputs(df):

    title = df["title"].fillna("").str.strip()
    body = df["text"].fillna("").str.strip().str.slice(0, TEXT_LIMIT)
    return (title + "\n" + body).tolist()


def embed(texts, url, model):

    payload = {"model": model, "input": texts, "encoding_format": "float"}
    for attempt in range(RETRIES):
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            items = sorted(r.json()["data"], key=lambda d: d["index"])
            return np.array([d["embedding"] for d in items], dtype=np.float32)
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            print(f"  ошибка запроса: {e}, повтор через {2 ** attempt} с")
            time.sleep(2 ** attempt)


def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0 
    return v / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DATA / "clean" / "news_2016.csv")
    ap.add_argument("--out", type=Path, default=DATA / "embeddings")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="USER-bge-m3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--append", action="store_true",
                    help="сохранить готовую матрицу и досчитать строки, дописанные в конец корпуса")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/embeddings"
    args.out.mkdir(parents=True, exist_ok=True)
    matrix_path = args.out / "matrix.npy"
    state_path = args.out / "state.json"

    df = pd.read_csv(args.src, dtype="string")
    missing = {"source", "dt_utc", "title", "text"} - set(df.columns)
    if missing:
        raise SystemExit(f"в {args.src} нет колонок: {sorted(missing)}")
    texts = build_inputs(df)
    total = len(texts)
    if total == 0:
        raise SystemExit(f"{args.src} пустой")
    print("статей:", total)


    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    done = state.get("done", 0) if state.get("total") == total else 0

    if args.append and not state and matrix_path.exists():
        old = np.load(matrix_path, mmap_mode="r")
        if len(old) > total:
            raise SystemExit(f"в матрице {len(old)} строк, а в корпусе {total} — дописывать нечего")
        if len(old) < total:

            tmp = matrix_path.with_name("matrix.tmp.npy")
            grown = np.lib.format.open_memmap(tmp, mode="w+", dtype=old.dtype, shape=(total, old.shape[1]))
            grown[:len(old)] = old
            grown.flush()
            done = len(old)
            del grown, old
            tmp.replace(matrix_path)
        else:
            done = total
            del old
        state_path.write_text(json.dumps({"total": total, "done": done}))
        matrix = np.lib.format.open_memmap(matrix_path, mode="r+")
        print("дописываем со строки", done)
    elif done and matrix_path.exists():
        matrix = np.lib.format.open_memmap(matrix_path, mode="r+")
        print("продолжаем со строки", done)
    else:
        done = 0
        dim = embed(texts[:1], url, args.model).shape[1]
        print("размерность:", dim)
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(total, dim))

    with tqdm(total=total, initial=done, unit="док") as bar:
        for i in range(done, total, args.batch):
            j = min(i + args.batch, total)
            matrix[i:j] = normalize(embed(texts[i:j], url, args.model))

            matrix.flush()
            state_path.write_text(json.dumps({"total": total, "done": j}))
            bar.update(j - i)

    pd.DataFrame({
        "row": np.arange(total),
        "article_id": [f"a_{i:07d}" for i in range(total)],
        "source": df["source"].to_numpy(),
        "dt_utc": df["dt_utc"].to_numpy(),
    }).to_csv(args.out / "id_map.csv", index=False)

    state_path.unlink(missing_ok=True)
    print(f"готово: {matrix_path} {matrix.shape}")


if __name__ == "__main__":
    main()
