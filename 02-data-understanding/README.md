# Data Understanding — DocForge

**Урок 4 · ML System Design**

Этот раздел документирует данные, на которых будет обучаться и валидироваться
ML-компонент DocForge — модуль Document Layout Analysis (DLA)

## ML-задача DocForge

DocForge подготавливает техническую документацию для корпоративных RAG-систем.
Внутри пайплайна — **Document Layout Analysis**: на странице PDF предсказать
bounding-box и класс каждого элемента (Title, Section-header, Text, List-item,
Table, Figure, Caption, Formula, Footnote, Page-header, Page-footer)

Качество DLA — ограничивающий фактор для всех downstream-этапов: восстановления
структуры, извлечения таблиц, семантического чанкинга, итогового результата в RAG.

**Метрика верхнего уровня:** mAP@0.5:0.95 на тестовом сете, per-class AP для критичных
классов (Title, Table, Section-header). Подробности — в [sampling_and_validation.md](sampling_and_validation.md).

## Датасет

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

### Почему DocLayNet
- Содержит сегмент **manuals** — тех. документация, целевой для DocForge
- COCO-формат — совместим с MMDetection / Detectron2 / Ultralytics
- **Двойная и тройная разметка** на части корпуса → честная оценка качества разметки
  и upper bound для ML (см. [data_quality.md](data_quality.md))
- На DocLayNet обучен **Docling** — production-бэкенд DocForge → бенчмарки прямо переносимы
- Стабильно поддерживается IBM, де-факто стандарт в индустрии DLA

### Чего нет в DocLayNet
- **Русского языка** — нужен внутренний RU-валидационный сет на 300–500 страниц
- **Сканов плохого качества** — компенсируем аугментациями (jpeg compression, blur, rotation)
- **Чертежей и схем** — для DocForge не критично на MVP, отложено в backlog

## Содержание раздела

| Файл | Содержание |
|---|---|
| [eda.ipynb](eda.ipynb) | Базовый EDA: распределения, дисбалансы, корреляции — с выводами для моделирования |
| [data_quality.md](data_quality.md) | Оценка качества разметки через inter-annotator agreement + предложения по улучшению |
| [sampling_and_validation.md](sampling_and_validation.md) | Алгоритм формирования выборки + 4-уровневая стратегия валидации |

## Ключевые выводы EDA

1. **Train/val/test зафиксированы** и сбалансированы по доменам, защищены от leakage → используем официальный split
2. **Manuals (26% корпуса)** — целевой домен → отдельный per-domain mAP в репортах
3. **Дисбаланс классов 100×** (Text 510k vs Title 5k) → focal loss + per-class AP
4. **Площади bbox от 0.1% до 75%** страницы → anchor-free детектор или 3-уровневые anchors
5. **Сильная ко-встречаемость** Caption↔Picture, Section-header↔Text → post-processing с rule-based reranking
6. **Только латиница** → собрать internal RU validation set
7. **Human upper bound ≈ 83% mAP** → целевой mAP DocForge MVP = 75% overall, ≥ 80% на Manuals

## Следующий этап (03-data-preparation)

1. Конвертация DocLayNet в формат выбранного фреймворка (MMDetection / Detectron2)
2. Подготовка subset «Manuals + Patents + Laws» для domain-targeted fine-tune
3. Сборка internal RU validation set (300–500 страниц)
4. Реализация augmentation pipeline (jpeg compression, blur, rotation, color jitter)
5. Скрипт автоматических проверок консистентности разметки

## Источники

- Pfitzmann B. et al. *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation.* KDD 2022. [arXiv:2206.01062](https://arxiv.org/abs/2206.01062)
- [DocLayNet GitHub](https://github.com/DS4SD/DocLayNet)
- [Docling — production бэкенд DocForge](https://github.com/DS4SD/docling)
- [HuggingFace ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet)
