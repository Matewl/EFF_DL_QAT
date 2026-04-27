# QAT для SASRec: сравнение методов квантизации

Сравнение 5 методов Quantization Aware Training на рекомендательной системе **SASRec** (Self-Attentive Sequential Recommendation).  
Датасет: **MovieLens-1M**. Метрика: **NDCG@10** на test-сплите.  
Лучший метод (**QDrop**) конвертируется в реальный INT8 и измеряется на CPU.

---

## Структура

```text
SasRec/
├── main.ipynb               # главный ноутбук: обучение + оценка
├── train_functions.py       # train_fp32 / train_qat / apply_adaround
├── bench.py                 # fake-quant CPU-бенчмарк
├── compare_results.py       # сравнительные графики по всем запускам
├── int8_conversion.py       # конвертация в real INT8 + бенчмарк
├── utils.py                 # load_config, ndcg_k, hit_k, ...
│
├── models/
│   ├── original.py          # SASRec (FP32)
│   └── quantization.py      # QuantSASRec + QuantMultiheadAttention
│
├── quantizations/
│   ├── quantizers.py        # FakeQuantizer, QuantLinear, QuantStrategy
│   ├── LSQ.py               # Learned Step-size Quantization
│   ├── APoT.py              # Additive Powers-of-Two
│   ├── QDrop.py             # Stochastic drop + uniform affine
│   ├── AdaRound.py          # Adaptive Rounding (PTQ)
│   └── utils.py
│
├── data/
│   ├── ml-1m.txt            # датасет MovieLens-1M
│   ├── dataloader.py        # create_dataloaders
│   ├── sasrec_dataset.py
│   └── utils.py             # load_movielens, data_partition
│
├── configs/
│   ├── base.yaml            # базовые гиперпараметры
│   ├── fp32.yml
│   ├── fp32_large.yml
│   ├── fp32_low_dropout.yml
│   ├── lsq.yml / lsq_asym.yml / lsq_4bit_asym.yml
│   ├── apot.yml / apot_perchannel.yml / apot_4bit.yml
│   ├── qdrop.yml / qdrop_highp.yml / qdrop_4bit.yml
│   └── adaround.yml / adaround_4bit.yml / adaround_v2.yml
│
├── checkpoints/
│   └── sasrec_runs/         # по папке на каждый запуск
│       ├── sasrec_fp32/
│       ├── sasrec_qdrop/    # ← используется для INT8
│       └── ...
│
└── results/
    ├── sasrec_quant_benchmark.json
    ├── sasrec_int8_comparison.json
    ├── comparison_best_per_method.png
    ├── comparison_all_runs.png
    ├── comparison_degradation.png
    ├── comparison_speed_vs_quality.png
    ├── comparison_training_curves.png
    └── sasrec_int8_comparison.png
```

---

## Установка

```bash
# из корня репозитория EFF_DL_QAT/
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision matplotlib numpy pandas tqdm pyyaml jupyter
```

Или активируйте готовое окружение:

```bash
source .venv/bin/activate
```

---

## Запуск

Все команды выполняются из директории `SasRec/`:

```bash
cd SasRec/
```

### Конфигурация модели (`configs/base.yaml`)

| Параметр | Значение |
|---|---|
| hidden_units | 64 |
| num_blocks | 2 |
| num_heads | 1 |
| maxlen | 50 |
| dropout_rate | 0.2 |
| Оптимизатор | Adam, lr=1e-3 |
| Эпохи | 20 |
| Batch size | 128 |

---

### 1. Ноутбук (рекомендуется)

```bash
jupyter notebook main.ipynb
```

Ноутбук содержит ячейки для каждого метода: FP32, LSQ, APoT, QDrop, AdaRound.

---

### 2. FP32 baseline

```python
from utils import load_config
from train_functions import Args, train_fp32
from models.quantization import QuantSASRec
from data.dataloader import create_dataloaders
import torch.nn as nn

config = load_config("configs/fp32.yml")
args   = Args(config)

train_loader, _, _, dataset = create_dataloaders(config)
[*_, usernum, itemnum] = dataset

model     = QuantSASRec(usernum, itemnum, args).to(args.device)
criterion = nn.BCEWithLogitsLoss()

train_fp32(
    model, train_loader, dataset, config, criterion, args,
    checkpoint_dir="checkpoints/sasrec_runs/sasrec_fp32",
    save_name="sasrec_fp32.pth",
)
```

---

### 3. QAT: LSQ / APoT / QDrop

Меняйте только имя конфига — всё остальное одинаково:

```python
config    = load_config("configs/qdrop_4bit.yml")   # qdrop.yml / lsq.yml / apot.yml
args      = Args(config)
quant_cfg = config["quantization"]
strategy  = quant_cfg["method"]   # "qdrop" | "lsq" | "apot"

train_loader, _, _, dataset = create_dataloaders(config)
[*_, usernum, itemnum] = dataset

model = QuantSASRec(usernum, itemnum, args).to(args.device)

# Опционально: загрузить FP32-чекпоинт для лучшей сходимости
# ckpt = torch.load("checkpoints/sasrec_runs/sasrec_fp32/sasrec_fp32.pth")
# model.load_state_dict(ckpt["model_state_dict"], strict=False)

train_qat(
    model, train_loader, dataset, config, criterion, args,
    strategy_name=strategy,
    quant_config=quant_cfg,
    checkpoint_dir=f"checkpoints/sasrec_runs/sasrec_{strategy}",
    save_name=f"sasrec_{strategy}.pth",
)
```

---

### 4. AdaRound (PTQ — без QAT-обучения)

Требует готового FP32-чекпоинта:

```python
config = load_config("configs/adaround.yml")
args   = Args(config)

train_loader, _, _, dataset = create_dataloaders(config)
[*_, usernum, itemnum] = dataset

model = QuantSASRec(usernum, itemnum, args).to(args.device)

apply_adaround(
    model, train_loader, dataset, args,
    fp32_checkpoint="checkpoints/sasrec_runs/sasrec_fp32/sasrec_fp32.pth",
    adaround_config=config["quantization"],
    checkpoint_dir="checkpoints/sasrec_runs/sasrec_adaround",
    save_name="sasrec_adaround.pth",
    config=config,
)
```

---

### 5. Fake-quant CPU-бенчмарк

```bash
CLEARML_DISABLE=1 python bench.py \
    --base-config configs/base.yaml \
    --results-out results/sasrec_quant_benchmark.json
```

---

### 6. Сравнительные графики

```bash
CLEARML_DISABLE=1 python compare_results.py
```

Генерирует в `results/`:

| Файл | Содержание |
|---|---|
| `comparison_best_per_method.png` | Лучший NDCG@10 / Hit@10 по каждому методу |
| `comparison_all_runs.png` | Все 16 запусков |
| `comparison_degradation.png` | % деградации от FP32 |
| `comparison_speed_vs_quality.png` | Throughput vs NDCG@10 |
| `comparison_training_curves.png` | Val NDCG по эпохам |

---

### 7. Real INT8: конвертация и бенчмарк

```bash
CLEARML_DISABLE=1 python int8_conversion.py
```

Результаты сохраняются в `results/sasrec_int8_comparison.json` и `results/sasrec_int8_comparison.png`.

---

## Результаты

### Лучший запуск по каждому методу (test split)

| Метод | Конфигурация | NDCG@10 | Hit@10 | Δ vs FP32 |
|---|---|---:|---:|---:|
| **QDrop** | **4-bit, p=0.5** | **0.01158** | **0.02235** | **+13.3%** |
| AdaRound | 8-bit | 0.01034 | 0.02086 | +1.2% |
| FP32 | baseline | 0.01022 | 0.02036 | — |
| APoT | 4-bit | 0.00986 | 0.02136 | −3.5% |
| LSQ | 8-bit asym | 0.00434 | 0.01043 | −57.5% |

---

### Все запуски (test NDCG@10)

| Run | Метод | Bits | Гиперпараметр | NDCG@10 | Hit@10 |
|---|---|---:|---|---:|---:|
| sasrec_fp32 | FP32 | — | — | 0.01022 | 0.02036 |
| sasrec_fp32_large | FP32 | — | hidden=128 | 0.00882 | 0.01854 |
| sasrec_fp32_low_dropout | FP32 | — | dropout=0.05 | 0.00637 | 0.01275 |
| **sasrec_qdrop_4bit** | **QDrop** | **4** | **p=0.5** | **0.01158** | **0.02235** |
| sasrec_qdrop | QDrop | 8 | p=0.5 | 0.01046 | 0.02152 |
| sasrec_qdrop_highp | QDrop | 8 | p=0.8 | 0.01026 | 0.02103 |
| sasrec_adaround | AdaRound | 8 | — | 0.01034 | 0.02086 |
| sasrec_adaround_v2 | AdaRound | 8 | v2 | 0.01034 | 0.02086 |
| sasrec_adaround_4bit | AdaRound | 4 | — | 0.01030 | 0.02086 |
| sasrec_apot_4bit | APoT | 4 | per-tensor | 0.00986 | 0.02136 |
| sasrec_apot | APoT | 8 | per-tensor | 0.00904 | 0.01937 |
| sasrec_apot_perchannel | APoT | 8 | per-channel | 0.00904 | 0.01937 |
| sasrec_lsq_asym | LSQ | 8 | asym | 0.00434 | 0.01043 |
| sasrec_lsq_run | LSQ | 8 | sym | 0.00399 | 0.00977 |
| sasrec_lsq_4bit_asym | LSQ | 4 | asym | 0.00334 | 0.00795 |
| sasrec_lsq | LSQ | 8 | sym | 0.00314 | 0.00778 |

---

### Fake-quant CPU-бенчмарк (val NDCG@10)

| Метод | NDCG@10 | Hit@10 | Avg latency, ms | Throughput, b/s | Size, MB |
|---|---:|---:|---:|---:|---:|
| **QDrop** | **0.01324** | **0.02815** | **68.7** | **14.6** | 3.56 |
| AdaRound | 0.01279 | 0.02731 | 70.8 | 14.1 | 1.38 |
| FP32 | 0.01279 | 0.02731 | 79.2 | 12.6 | 3.54 |
| APoT | 0.00706 | 0.01507 | 77.0 | 13.0 | 3.55 |
| LSQ | 0.00449 | 0.01010 | 75.0 | 13.3 | 3.57 |

---

### Real INT8 vs FP32 (QDrop 8-bit → dynamic INT8, CPU)

Метод: `torch.quantization.quantize_dynamic({nn.Linear}, dtype=qint8)`, backend `qnnpack`.

#### Качество

| Метрика | FP32 | INT8 | Δ |
|---|---:|---:|---:|
| NDCG@10 (val) | 0.01207 | 0.01207 | 0.0% |
| Hit@10 (val) | 0.02533 | 0.02533 | 0.0% |
| NDCG@10 (test) | 0.01089 | 0.01074 | −1.4% |
| Hit@10 (test) | 0.02301 | 0.02252 | −2.2% |

#### Скорость (CPU, batch_size=128)

| Метрика | FP32 | INT8 | Speedup |
|---|---:|---:|---:|
| Avg latency, ms | 40.8 | 42.4 | 0.96× |
| Median latency, ms | 41.4 | 42.7 | 0.97× |
| Throughput, батчей/с | 24.5 | 23.6 | 0.96× |
| Размер параметров | 1.17 MB | 1.04 MB | −10.8% |

> **Вывод:** Качество практически не изменилось (−1.4% NDCG).  
> Ускорения на CPU нет: модель слишком маленькая (hidden=64), Linear-слои занимают ~0.3 MB
> из 1.17 MB — остальное Embedding-таблицы, которые dynamic INT8 не трогает.
> Overhead на квантизацию активаций перевешивает выигрыш от INT8 GEMM.
> Speedup ожидается при hidden_units ≥ 256.

---

### Ключевые наблюдения

1. **QDrop — лучший метод.** Stochastic quantization drop работает как регуляризатор и позволяет модели обойти FP32 baseline (+13% NDCG). 4-bit QDrop лучше 8-bit: меньшая битность усиливает регуляризационный эффект.

2. **AdaRound — стабильный PTQ.** Не требует QAT-обучения, результат близок к FP32 (+1.2%). Самый маленький чекпоинт (1.38 MB против 3.5 MB у остальных — чекпоинт хранит только округлённые веса, без оптимизатора).

3. **APoT — умеренная деградация.** Per-channel не дал преимущества перед per-tensor. 4-bit лучше 8-bit (аналогично QDrop).

4. **LSQ нестабилен на маленьких моделях.** Обучаемый step size расходится при hidden=64. Лучший вариант (8-bit asym) теряет 57% NDCG относительно FP32.

5. **Real INT8 не ускоряет маленькую модель.** Dynamic quantization выгодна при больших Linear-слоях. Для SASRec с hidden=64 доминирует Embedding lookup и overhead квантизации активаций.
