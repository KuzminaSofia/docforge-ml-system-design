# Data Understanding — DocForge

**Урок 4 · ML System Design**

Этот раздел документирует данные, на которых будет обучаться и валидироваться
ML-компонент DocForge — модуль Document Layout Analysis (DLA).

Базовый EDA с графиками и кодом — в [eda.ipynb](eda.ipynb).
Этот файл — текстовое сопровождение: источник данных, оценка качества разметки,
стратегия выборки и валидации.

---

## 1. ML-задача DocForge

DocForge подготавливает техническую документацию для корпоративных RAG-систем.
Внутри пайплайна — **Document Layout Analysis**: на странице PDF предсказать
bounding-box и класс каждого элемента (Title, Section-header, Text, List-item,
Table, Figure, Caption, Formula, Footnote, Page-header, Page-footer).

Качество DLA — ограничивающий фактор для всех downstream-этапов: восстановления
структуры, извлечения таблиц, семантического чанкинга, итогового результата в RAG.

**Метрика верхнего уровня:** mAP@0.5:0.95 на тестовом сете, per-class AP для критичных
классов (Title, Table, Section-header). См. раздел 5 ниже.

---

## 2. Источник и состав данных

**[DocLayNet](https://github.com/DS4SD/DocLayNet)** (IBM Research, 2022)

| | |
|---|---|
| Размер | 80 863 страницы PDF |
| Классы | 11 |
| Домены | 6 (financial_reports, manuals, scientific_articles, laws_and_regulations, government_tenders, patents) |
| Языки | EN (~92%), DE, FR, JA |
| Формат | COCO bounding-boxes + PNG (1025×1025) + исходные PDF |
| Splits | train 69 375 / val 6 489 / test 4 999 (document-level, без leakage) |
| Лицензия | CDLA-Permissive 1.0 — коммерческое использование разрешено |
| Источник | [HuggingFace ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet) или [IBM Cloud (zip)](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip) |
| Объём | 28 GiB (core) + 7.5 GiB (extra: PDFs и cell-level JSONs) |

### Почему именно DocLayNet
- Содержит сегмент **manuals** — тех. документация, целевой домен DocForge
- COCO-формат — совместим с MMDetection / Detectron2 / Ultralytics
- **Двойная и тройная разметка** на части корпуса → честная оценка качества разметки
  и upper bound для ML
- На DocLayNet обучен **Docling** — production-бэкенд DocForge → бенчмарки прямо переносимы
- Стабильно поддерживается IBM, де-факто стандарт в индустрии DLA

### Чего нет в DocLayNet
- **Русского языка** — нужен внутренний RU-валидационный сет на 300–500 страниц
- **Сканов плохого качества** — компенсируем аугментациями (jpeg compression, blur, rotation)
- **Чертежей и схем** — для DocForge не критично на MVP, отложено в backlog

---

## 3. Базовый EDA — ключевые выводы

Графики и код в [eda.ipynb](eda.ipynb). Здесь — выжимка для моделирования.

| # | Наблюдение | Действие в моделировании |
|---|---|---|
| 1 | Train/val/test зафиксированы и сбалансированы по доменам | Использовать **официальный split** без перемешивания |
| 2 | Manuals = 26% корпуса, целевой для DocForge | Отчётность **per-domain mAP**, особенно на manuals |
| 3 | Дисбаланс классов 100× (Text 510k vs Title 5k) | **focal loss** + per-class AP в репортах |
| 4 | Площади bbox от 0.1% до 75% страницы | Anchor-free детектор или 3-уровневые anchors |
| 5 | Сильные ко-встречаемости (Caption↔Picture, Section-header↔Text) | Post-processing с rule-based reranking |
| 6 | Только латиница (EN/DE/FR/JA); RU нет | Собрать internal RU validation set; fine-tune под кириллицу |
| 7 | Human upper bound ≈ 83% mAP | Целевой mAP DocForge MVP = **75% overall, ≥ 80% на manuals** |

---

## 4. Оценка качества разметки

### 4.1. Источник оценки

DocLayNet содержит **redundantly annotated subset** — часть страниц размечена несколькими экспертами независимо. Это даёт прямую измеримую оценку качества разметки через inter-annotator agreement (IAA).

- Double-annotated: **7 059 страниц** (~ 8.7% корпуса)
- Triple-annotated: **1 374 страницы** (~ 1.7% корпуса)
- Метрика согласованности: **mAP@0.5:0.95** между парами аннотаторов

### 4.2. Результаты по классам

| Класс | Human mAP, % | Оценка | Причина расхождений |
|---|---|---|---|
| Formula | 100 | отличное | чёткие визуальные признаки (LaTeX-блоки) |
| Page-footer | 89 | хорошее | стабильное положение и шрифт |
| List-item | 88 | хорошее | bullet-маркеры, отступы |
| Page-header | 86 | хорошее | стабильное положение |
| Caption | 84 | приемлемое | граница caption / body неоднозначна |
| Text | 84 | приемлемое | склейка соседних абзацев — частый спор |
| Footnote | 83 | приемлемое | граница footnote / page-footer |
| Table | 82 | приемлемое | границы крупных таблиц с подписями |
| Section-header | 79 | посредственное | граница Title / Section-header |
| Picture | 71 | слабое | диаграммы vs формулы, сложные сцены |
| **Title** | **60** | **слабое** | пограничный класс с Section-header |

**Средний human mAP по корпусу ≈ 83%.**

### 4.3. Интерпретация для DocForge

- **Объективный потолок качества.** Никакая ML-модель не сможет получить mAP выше ≈ 83% — это не баг, а естественное свойство задачи (некоторые границы объективно неоднозначны). Цели вида «≥ 95% качества» нереалистичны.
- **Самые проблемные классы критичны для продукта.** Title (60%) — первый чанк документа в RAG, ошибки прямо в Knowledge Base. Section-header (79%) — определяет иерархию чанков. Picture (71%) — без правильной локализации в чанки попадает мусор.
- **Качество разметки слабее зависит от объёма данных**, чем от объективной сложности границ. Title и Formula оба редкие, но согласованность 60% vs 100%.

### 4.4. Предложения по улучшению (для внутреннего RU-сета)

Если бы мы собирали внутреннюю русскоязычную разметку, заложили бы:

1. **Уточнённая labeling guideline** — явно прописать пограничные случаи: Title vs Section-header, Caption vs Text под рисунком, Picture vs Formula, Footnote vs Page-footer с примерами.
2. **Cross-annotation на «горячих» классах** — Title, Section-header, Picture, Caption обязательно двойная разметка + arbitration третьим экспертом при IoU < 0.7 или class mismatch.
3. **Active learning loop** — после обучения на 1000 страниц модель показывает аннотаторам страницы с наибольшей неопределённостью; разметка идёт именно на них — выигрыш в качестве при меньшем объёме.
4. **Автопроверки консистентности** — bbox не пересекаются больше чем на 30% IoU; сумма площадей ≤ 0.95 страницы; на странице ровно один Title; Caption — только рядом с Picture/Table. Не прошедшие — на ручную верификацию.
5. **Версионирование labeling guideline** — любое изменение → перепроверка ранее размеченного на затронутых классах.

### 4.5. Что делаем сейчас

DocLayNet используем как есть (это open-source бенчмарк, модифицировать нельзя), но:
- Метрика — **per-class AP**, а не только среднее mAP
- В test-репорт — **error analysis по 50 случайным ошибкам на Title**
- Параллельно готовим internal RU-валидационный сет по guideline выше (≥ 300 страниц)

---

## 5. Алгоритм формирования выборки и стратегия валидации

### 5.1. Исходные данные

DocLayNet поставляется с **предопределёнными splits** от авторов:

| Split | Страниц | Доля |
|---|---|---|
| train | 69 375 | 85.8% |
| validation | 6 489 | 8.0% |
| test | 4 999 | 6.2% |

Split сделан **на уровне документов, не страниц**: страницы одного исходного PDF целиком уходят в один сплит. Это критично — иначе разные страницы одного отчёта попали бы в train и test (data leakage, модель «видит» похожие лейауты).

**Решение: используем официальный split без изменений.** Он сбалансирован по доменам и защищён от leakage.

### 5.2. Формирование выборки для обучения

**Базовый сценарий — full corpus training.** Train (69 375 страниц) используется целиком — максимальное разнообразие лейаутов.

**Альтернатива — domain-targeted subset.** Filter по `doc_category in {manuals, patents, laws_and_regulations}` ≈ 26 000 страниц. Быстрее обучение, ближе к продакшен-распределению, но риск хуже обобщаться на «нестандартные» лейауты.

**Решение:** обучаем **две модели** — full и domain-targeted — и сравниваем на validation. Выбираем ту, что выше на manuals.

**Балансировка классов — focal loss (γ = 2.0)**. Это стандарт для COCO-style детекции и не требует менять loader. Альтернативы (class-balanced sampling, repeat factor sampling) усложняют батчинг без явного выигрыша.

**Аугментации:**

| Аугментация | Зачем | Параметры |
|---|---|---|
| Random resize | устойчивость к разрешению | scale 0.5–1.5 |
| Random crop | устойчивость к обрезке | min IoU 0.5 |
| Color jitter | сканы разного качества | brightness ±0.3, contrast ±0.3 |
| Gaussian blur | низкокачественные PDF | sigma 0–2 |
| JPEG compression | эмуляция сканов | quality 50–95 |
| Rotation ±2° | сканы под лёгким наклоном | вероятность 0.3 |
| Horizontal flip | **НЕ применяем** | flip ломает порядок чтения (LTR) |

### 5.3. Стратегия валидации — 4 уровня

**Уровень 1 — Validation (in-loop).** После каждой epoch на val (6 489 страниц): overall mAP@0.5:0.95, per-class AP, per-domain mAP (особенно manuals), loss. Early stopping: patience 5 epoch по overall mAP.

**Уровень 2 — Test (финальная оценка).** Test (4 999 страниц) трогаем **один раз** перед коммитом в продакшен. Не используем для tuning гиперпараметров. Дополнительно: per-class AP с bootstrap CI (1000 ресэмплов) + error analysis на 50 худших примерах.

**Уровень 3 — Internal RU validation (production-fit).** DocLayNet — латиничный. DocForge продаётся в РФ. Нужен внутренний RU-сет: 300–500 страниц русских PDF (ГОСТы, тех. регламенты, мануалы, паспорта оборудования), двойная разметка по той же 11-классной схеме, IoU 0.7 порог согласия, arbitration при конфликте. Это финальный gate перед релизом: если mAP падает > 10 п.п. относительно DocLayNet test → нужен domain adaptation.

**Уровень 4 — Production shadow mode.** Первые 2 недели после релиза модель работает в shadow: предсказывает, но результат сравнивается с текущим Docling backend. Расхождения логируются.

### 5.4. Защита от типичных ошибок

| Ошибка | Защита |
|---|---|
| Leakage train ↔ test | document-level split от авторов |
| Cherry-picking на test | test трогаем один раз; tuning только на val |
| Smoothed mAP скрывает деградации редких классов | per-class AP в репортах обязательно |
| Overfit на доминирующий домен (financial) | per-domain mAP в репорте, ранний стоп по manuals |
| Дрифт после релиза | shadow mode + еженедельный re-evaluation |

### 5.5. Метрики приёмки модели

Модель допускается в production, если **все** условия выполнены на test set DocLayNet:

| Метрика | Порог |
|---|---|
| Overall mAP@0.5:0.95 | ≥ 0.70 |
| Per-class AP — Title | ≥ 0.55 |
| Per-class AP — Table | ≥ 0.75 |
| Per-class AP — Section-header | ≥ 0.65 |
| mAP на manuals | ≥ 0.72 |

**Дополнительно** на internal RU validation:

| Метрика | Порог |
|---|---|
| Overall mAP@0.5:0.95 | ≥ 0.60 |
| F1 по Text + Section-header | ≥ 0.75 |

Хотя бы один порог не пройден — модель не релизим, идём в error analysis и доcборку данных.

### 5.6. Cross-validation: нужна или нет?

Для **DocLayNet** — нет. 80k страниц + фиксированные splits → k-fold не даст более точной оценки, чем единичный test, а вычислительно дороже в k раз.

Для **internal RU set** (300–500 страниц) — да, **stratified 5-fold по доменам**. Маленький объём → одна выборка test даст шумную оценку.

---

## 6. Следующий этап (03-data-preparation)

1. Конвертация DocLayNet в формат выбранного фреймворка (MMDetection / Detectron2)
2. Подготовка subset «Manuals + Patents + Laws» для domain-targeted fine-tune
3. Сборка internal RU validation set (300–500 страниц)
4. Реализация augmentation pipeline
5. Скрипт автоматических проверок консистентности разметки

---

## Источники

- Pfitzmann B. et al. *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation.* KDD 2022. [arXiv:2206.01062](https://arxiv.org/abs/2206.01062)
- [DocLayNet GitHub](https://github.com/DS4SD/DocLayNet)
- [Docling — production бэкенд DocForge](https://github.com/DS4SD/docling)
- [HuggingFace ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet)
