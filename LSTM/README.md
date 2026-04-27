# QAT для LSTM: сравнение методов квантизации

Сравнение 5 методов QAT & PTQ на **LSTM** для задачи классификации текста.
Датасет: **Yelp Polarity** (150k train / 15k test сплит). Метрика: **ROCAUC** на валидационном датасете.
Лучший метод (PACT) конвертируется в реальный INT8 и измеряется на CPU.

## Структура

```text
EFF_DL_QAT/
├── LSTM/
│   ├── Main_Experiments.ipynb    # Главный ноутбук: обучение всех методов, графики, замер метрик
│   └── Main_Experiments.py       # Консольная версия пайплайна
│
├── src/
│   ├── data.py                   # Загрузка Yelp Polarity, токенизация, создание DataLoader
│   ├── engine.py                 # Логика обучения, сохранение лучших весов
│   ├── model.py                  # FakeQuantizedLSTM, обертки слоев и CustomQAT_LSTM
│   └── quantization.py           # LSQ, PACT, APoT, DSQ, AdaRound + Parametrization
│
├── results/
│   ├── lstm_fp32_best.pt         # Базовые веса
│   ├── lstm_pact_best.pt         # Веса лучшей QAT модели
│   ├── lstm_apot_best.pt         # И т.д. для каждого метода
│   └── roc_auc_final_comparison.png # Итоговый график сравнения обучения
│
├── .gitignore
└── README.md
```

## Установка

Из корня репозитория создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate        # Для Windows: .venv\Scripts\activate
```

Установите необходимые зависимости:

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn datasets jupyter
```

## Запуск

Все основные эксперименты собраны в единый пайплайн в Jupyter Notebook. 
Запустите Jupyter из корневой папки:

```bash
jupyter notebook LSTM/Main_Experiments.ipynb
```

**Альтернатива** Запуск полного цикла тестирования в консоли:
```bash
python "LSTM/Main_Experiments.py"
```

## Как это работает

1. **Базовая модель (FP32):** Обучается стандартная двунаправленная LSTM с параметрами `embedding_dim=128`, `hidden_dim=128`.
2. **QAT Цикл:** Граф вычислений модифицируется с помощью `torch.nn.utils.parametrize`. Методы квантизации применяются к весам LSTM, Embedding-слою и полносвязному слою.
3. **Запекание (Baking):** Обученные QAT-веса извлекаются из оберток. Кастомные слои трансформируются в нативные слои `nn.Embedding` и `nn.Linear`.
4. **Конвертация в INT8:** Итоговая модель прогоняется через `torch.ao.quantization.quantize_dynamic` для реального сжатия.

---

## Результаты

### Итоговые метрики Fake Quantization

Ниже представлен топ методов по пиковому значению ROC-AUC на валидационной выборке:

| Метод | ROC-AUC | Δ vs FP32 |
| :--- | :--- | :--- |
| **FP32** | **0.9844** | — |
| **PACT** | **0.9834** | -0.10% |
| **APoT** | **0.9828** | -0.16% |
| **LSQ** | **0.9816** | -0.28% |
| **DSQ** | **0.9800** | -0.44% |
| **AdaRound** | **0.9451** | -3.99% |

### Real INT8 PACT vs FP32 (CPU)

| Метрика | FP32 | INT8 (PACT) | Изменение |
| :--- | :--- | :--- | :--- |
| **ROCAUC (test)** | 0.9849 | 0.9843 | **0.0006** |
| **Размер модели** | 10.84 MB | 2.92 MB | **Сжатие 3.71x** |
| **Avg Latency (ms)** | 184.62 ms | 164.02 ms | **Ускорение 1.13x** |

---
