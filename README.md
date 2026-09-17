# tailsGPT

Небольшая GPT-подобная языковая модель для генерации сказок.

Модель реализована с нуля на PyTorch и обучается на корпусе сказок братьев Гримм из Project Gutenberg. Полный цикл построения decoder-only Transformer: подготовку корпуса, обучение BPE-токенизатора, обучение модели и авторегрессионную генерацию текста.

## Идея

Во время генерации этот процесс повторяется авторегрессионно:

```text
Начальный текст
      │
      ▼
BPE-токенизация
      │
      ▼
Decoder-only Transformer
      │
      ▼
Распределение вероятностей следующего токена
      │
      ▼
Sampling
      │
      ▼
Новый токен
      │
      └──────────────► повтор
```

За счет обучения на одном литературном корпусе модель воспроизводит характерную лексику и структуру классических сказок.

## Архитектура

Модель построена по decoder-only Transformer-схеме:

```text
Token Embedding
       +
Position Embedding
       │
       ▼
┌─────────────────────┐
│ Transformer Block   │
│                     │
│ LayerNorm           │
│ Multi-Head Attention│
│ Residual Connection │
│ LayerNorm           │
│ Feed Forward        │
│ Residual Connection │
└─────────────────────┘
          × 4
       │
       ▼
   LayerNorm
       │
       ▼
 Linear LM Head
       │
       ▼
Vocabulary logits
```

Основные параметры:

```text
Vocabulary size:    5000
Context length:     256
Embedding size:     384
Attention heads:    6
Transformer layers: 4
Dropout:            0.2
```

В attention используется causal mask, поэтому каждый токен имеет доступ только к предыдущей части последовательности.

## Подготовка данных

Корпус автоматически загружается из Project Gutenberg.

Перед обучением:

1. из исходного файла выделяется текст сказок;
2. обучается собственный BPE-токенизатор;
3. текст преобразуется в последовательность token IDs;
4. данные делятся на train и validation в пропорции 90/10.

Таким образом модель и токенизатор обучаются на одном и том же корпусе без использования pretrained-моделей.

## Структура проекта

```text
tails-gpt/
├── config.py            # параметры модели и обучения
├── data.py              # загрузка корпуса и обучение BPE
├── model.py             # реализация Transformer
├── train.py             # обучение модели
├── generate.py          # генерация текста
│
├── fairy_tales.txt      # исходный корпус
├── temp_corpus.txt      # корпус для обучения tokenizer
├── fairy_tale_gpt.pth   # веса обученной модели
│
├── requirements.txt
└── README.md
```

## Обучение

Установка зависимостей:

```bash
git clone https://github.com/osmanof/tails-gpt.git
cd tails-gpt

pip install -r requirements.txt
```

Запуск обучения:

```bash
python train.py
```

Используется оптимизатор `AdamW` и cross-entropy loss для предсказания следующего токена.

После обучения веса сохраняются в:

```text
fairy_tale_gpt.pth
```

## Генерация

Для генерации текста:

```bash
python generate.py
```

По умолчанию используется начало:

```text
Once upon a time, in a dark forest, there lived a
```

Модель последовательно генерирует продолжение, выбирая следующий токен из полученного распределения вероятностей.

Пример:

```text
Once upon a time, in a dark forest, there lived a ...
```

Начальный текст и длину генерации можно изменить в `generate.py`.

## Стек

* Python
* PyTorch
* Hugging Face Tokenizers
* BPE
* Transformer
* Causal Self-Attention
