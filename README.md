Установка vLLM:
```
pip install vllm
```

Команда для pod модель: USER-bge-m3
```
vllm serve deepvk/USER-bge-m3 --served-model-name USER-bge-m3 --runner pooling --max-model-len 8192 --gpu-memory-utilization 0.70 --host 127.0.0.1 --port 8000
```

Команда для pod модель: RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic
```
vllm serve RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic --served-model-name Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.85 --host 127.0.0.1 --port 8002
```

# Анализ новостей 2016

Данные должны лежать в `../data` (рядом с папкой проекта), исходные архивы в `data/raw/`.

# Установка среды:
```
uv sync
```



## Порядок запуска

* `analysis.ipynb`- осмотр архивов, выгрузка 2016, `merged.csv` 
* `cleaning.ipynb`- P1, очистка -> `clean/news_2016.csv` 
* `uv run python embed.py`- P2, эмбеддинги статей 
* `dedup_stories.ipynb` - P3-P4, дубликаты и сюжеты -> `stories/` 
* `uv run python calibrate.py --note "..."` - P0, проверка схемы и промпта на 100 статьях 
* `uv run python enrich.py` - P5, обогащение LLM -> `enriched/` 
* `uv run python postprocess.py` - P6, сущности -> `entities/`, нужен `OPENAI_API_KEY` 
* `uv run python topics.py`- темы (BERTopic) 
* `uv run python trends.py` - тренды 
* `uv run python events.py` -всплески и события 
* `uv run python entity_graph.py` - граф сущностей 


Отчёты и логи пишутся в `data/reports/`.