# ESPCN

Текущая рабочая часть проекта посвящена модели **ESPCN** для single image super-resolution и сравнению качества/скорости разных вариантов квантизации.

Основной код находится в [`ESPCN`](ESPCN):

- [`model.py`](model.py) - базовая ESPCN-модель.
- [`dataset.py`](dataset.py) - датасеты DIV2K, Set5 и Set14.
- [`train.py`](train.py) - CLI для обучения, тестирования и запуска квантизационных экспериментов.
- [`quantizations`](quantizations) - реализации PACT, LSQ, APoT, AdaRound и QDrop.
- [`int8_convertation.py`](int8_convertation.py) - утилиты для конвертации PACT-модели в real INT8 и CPU-бенчмарка.

## Установка

Рекомендуется запускать проект из корня проекта ESPCN.

```bash
cd ESPCN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Данные

По умолчанию код ожидает данные в `data`.

```text
data/
├── DIV2K_train_HR/
├── DIV2K_valid_HR/
├── Set5/
│   ├── GTmod12/
│   ├── LRbicx2/
│   ├── LRbicx3/
│   └── LRbicx4/
└── Set14/
    ├── GTmod12/
    ├── LRbicx2/
    ├── LRbicx3/
    └── LRbicx4/
```

`DIV2K_train_HR` и `DIV2K_valid_HR` используются для обучения и валидации. Для тестирования используются Set5 и Set14 с bicubic LR-изображениями для масштабов `x2`, `x3`, `x4`.

## Быстрый Запуск

Базовый FP32-эксперимент для `x2`:

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --epochs 150 \
  --batch-size 64 \
  --quant-method wo_quant \
  --experiment-name wo_quant_x2
```

Логи и чекпоинты сохраняются в `runs/<experiment-name>/version_*/`. Лучший чекпоинт выбирается по `val/psnr`.

Для просмотра TensorBoard:

```bash
tensorboard --logdir runs
```

## Аргументы `train.py`

Основные параметры:

| Аргумент | Описание | Значение по умолчанию |
| --- | --- | --- |
| `--data-root` | Путь к данным | `data` |
| `--scale` | Коэффициент super-resolution | `2`; варианты: `2`, `3`, `4` |
| `--patch-size` | Размер LR-патча для обучения | `32` |
| `--batch-size` | Batch size для обучения | `16` |
| `--epochs` | Количество эпох | `100` |
| `--lr` | Learning rate | `1e-3` |
| `--weight-decay` | Weight decay | `0.0` |
| `--quant-method` | Метод квантизации | `wo_quant`; варианты: `wo_quant`, `pact`, `lsq`, `adaround`, `apot`, `qdrop` |
| `--act-bits` | Битность активаций | `8` |
| `--weight-bits` | Битность весов | `8` |
| `--checkpoint-path` | Путь к `.ckpt` для теста или PTQ | `None` |
| `--test-only` | Только тестирование чекпоинта | выключено |
| `--logger` | Логгер | `tensorboard`; варианты: `tensorboard`, `csv` |

## Эксперименты

### 1. FP32 Baseline

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --epochs 150 \
  --batch-size 64 \
  --quant-method wo_quant \
  --experiment-name wo_quant_x2
```

### 2. PACT QAT

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --epochs 150 \
  --batch-size 64 \
  --quant-method pact \
  --act-bits 8 \
  --weight-bits 8 \
  --experiment-name pact_w8a8_x2
```

Для signed-активаций:

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --epochs 150 \
  --batch-size 64 \
  --quant-method pact \
  --signed \
  --act-bits 8 \
  --weight-bits 8 \
  --experiment-name pact_signed_w8a8_x2
```

### 3. LSQ QAT

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --epochs 150 \
  --batch-size 64 \
  --quant-method lsq \
  --act-bits 8 \
  --weight-bits 8 \
  --experiment-name lsq_w8a8_x2
```

### 4. APoT

В репозитории есть реализация APoT-квантизатора в [`quantizations/apot.py`](quantizations/apot.py), а CLI содержит опцию `--quant-method apot`.

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --epochs 150 \
  --batch-size 64 \
  --quant-method apot \
  --act-bits 8 \
  --weight-bits 8 \
  --experiment-name apot_w8a8_x2
```

### 5. AdaRound PTQ

AdaRound запускается поверх уже обученного FP32-чекпоинта. Сначала обучите baseline, затем передайте путь к лучшему чекпоинту:

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --quant-method adaround \
  --checkpoint-path runs/wo_quant_x2/version_0/checkpoints/best-XX.ckpt \
  --experiment-name adaround_x2
```

Замените `best-XX.ckpt` на реальное имя чекпоинта из вашей папки `checkpoints`.

### 6. QDrop PTQ/Fine-tuning

QDrop также стартует с обученного чекпоинта:

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --quant-method qdrop \
  --checkpoint-path runs/wo_quant_x2/version_0/checkpoints/best-XX.ckpt \
  --experiment-name qdrop_x2
```

В текущей реализации QDrop дообучается внутри [`train_q_drop`](quantizations/qdrop.py) на 150 эпохах с `lr=1e-5`.

### 7. Тестирование Готового Чекпоинта

```bash
python train.py \
  --data-root data \
  --scale 2 \
  --test-only \
  --checkpoint-path runs/pact_w8a8_x2/version_0/checkpoints/best-XX.ckpt
```

Скрипт печатает PSNR и SSIM отдельно для Set5 и Set14.

### 8. Эксперименты Для Других Scale

Для `x3` или `x4` поменяйте `--scale` и имя эксперимента:

```bash
python train.py \
  --data-root data \
  --scale 4 \
  --epochs 150 \
  --batch-size 64 \
  --quant-method pact \
  --act-bits 8 \
  --weight-bits 8 \
  --experiment-name pact_w8a8_x4
```

## Real INT8 И CPU-Бенчмарк

[`int8_convertation.py`](int8_convertation.py) содержит функции для:

- выгрузки PACT-модели в FP32-структуру;
- подготовки `torch.ao.quantization` модели с `QuantStub`/`DeQuantStub`;
- калибровки;
- сравнения PSNR/SSIM;
- CPU-бенчмарка FP32 и INT8.


### Качество Super-Resolution

| Эксперимент | Scale | Метод | W/A bits | Signed | Set5 PSNR | Set5 SSIM | Set14 PSNR | Set14 SSIM | Checkpoint | Комментарий |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `wo_quant_x2` | x2 | FP32 | - | - |  |  |  |  |  |  |
| `pact_w8a8_x2` | x2 | PACT | 8/8 | no |  |  |  |  |  |  |
| `lsq_w8a8_x2` | x2 | LSQ | 8/8 | no |  |  |  |  |  |  |
| `adaround_x2` | x2 | AdaRound | 8/- | - |  |  |  |  |  |  |
| `qdrop_x2` | x2 | QDrop | 8/8 | - |  |  |  |  |  |  |

### Скорость INT8

| Эксперимент | Dataset | Backend | Threads | FP32 latency, ms/img | INT8 latency, ms/img | FP32 FPS | INT8 FPS | Speedup | PSNR drop | SSIM drop |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pact_w8a8_x2_int8` | Set5 | `x86`/`qnnpack` |  |  |  |  |  |  |  |  |
| `pact_w8a8_x2_int8` | Set14 | `x86`/`qnnpack` |  |  |  |  |  |  |  |  |

### Параметры Запуска

| Эксперимент | Seed | Epochs | Batch size | LR | Weight decay | Patch size | Accelerator | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `wo_quant_x2` | 42 | 150 | 64 | 1e-3 | 0.0 | 32 | `auto` | `32-true` |
| `pact_w8a8_x2` | 42 | 150 | 64 | 1e-3 | 0.0 | 32 | `auto` | `32-true` |

## Структура Репозитория

```text
.
├── README.md
├── ef_models.pdf
└── 
    ├── data/
    ├── dataset.py
    ├── int8_convertation.py
    ├── lightning_module.py
    ├── model.py
    ├── quantizations/
    ├── run_all.ipynb
    ├── train.py
    └── utils.py
```

## Участники

- **Лузанин Матвей** - @luzaninmatvey
- **Бурминов Михаил** - @evolvens
- **Мартиросян Завен** - @zmazak0
