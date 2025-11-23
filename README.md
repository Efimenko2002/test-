# Обучение Faster R-CNN в Google Colab

Этот репозиторий содержит пример кода для обучения модели обнаружения объектов (Faster R-CNN) на собственном датасете с разметкой в формате YOLO (`class x_center y_center width height`). Код ориентирован на запуск в Google Colab и работу с датасетом, размещённым в Google Drive.

## Структура датасета

В каталоге с данными каждая картинка `*.png` сопровождается файлом `*.txt` с тем же именем. В файле разметки каждая строка описывает один объект:

```
<class_id> <x_center> <y_center> <width> <height>
```

* `class_id` — целое число от `0` для первого класса; фон добавляется автоматически.
* `x_center`, `y_center`, `width`, `height` — нормализованные координаты относительно ширины и высоты изображения.

## Быстрый старт в Colab

1. Используйте готовый ноутбук `colab_faster_rcnn.ipynb` (загрузите его в Colab или откройте напрямую в Google Drive) и выполните ячейки последовательно.
2. В первой кодовой ячейке (или в дополнительной) установите зависимости:

```python
!pip install torchmetrics==1.4.0
```

3. Смонтируйте Google Drive и укажите путь к каталогу с данными:

```python
from google.colab import drive
from colab_faster_rcnn import TrainConfig, run_training

drive.mount('/content/drive')
DATA_DIR = '/content/drive/MyDrive/path/to/dataset'
```

4. Запустите обучение (параметры можно менять):

```python
cfg = TrainConfig(
    data_dir=DATA_DIR,
    output_dir='/content/drive/MyDrive/rcnn_runs',
    num_epochs=5,
    batch_size=2,
)
model, history, test_metrics = run_training(cfg)
```

5. После обучения в `output_dir` появятся:

* `model.pt` — веса модели и конфигурация.
* `training_curves.png` — график снижения функции потерь и качества (mAP) на валидации.

## Замечания

* Модель использует предобученные веса `fasterrcnn_resnet50_fpn` из `torchvision`, поэтому в Colab достаточно стандартной установки PyTorch.
* Для расчёта метрик используется `torchmetrics`. Если библиотека недоступна, выводится предупреждение и можно установить пакет вручную.
* При желании можно менять разбиение датасета (поля `train_ratio`/`val_ratio` в `TrainConfig`) и гиперпараметры обучения.

