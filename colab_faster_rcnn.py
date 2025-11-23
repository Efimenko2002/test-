"""
Пример кода для обучения Faster R-CNN в Google Colab.
Скрипт ожидает датасет в формате: для каждого изображения .png существует .txt
cо строками YOLO: <class_id> <x_center> <y_center> <width> <height>, все координаты нормализованы.
Каталоги должны быть заранее разбиты на train/val/test c подпапками images/ и labels/.
"""
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Опциональные зависимости для метрик и графиков
try:
    import torchmetrics
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except Exception:  # noqa: BLE001
    torchmetrics = None
    MeanAveragePrecision = None

import matplotlib.pyplot as plt
from PIL import Image


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "runs"
    batch_size: int = 2
    num_epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2


class YoloTxtDataset(Dataset):
    """
    Кастомный датасет для пар .png/.txt с YOLO-аннотациями.
    Возвращает изображение тензор и словарь target, совместимый с torchvision detection API.
    """

    def __init__(self, samples: List[Tuple[str, str]]):
        self.samples = samples
        self.transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32)])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path, txt_path = self.samples[idx]
        image = read_image(img_path)
        c, h, w = image.shape

        boxes: List[List[float]] = []
        labels: List[int] = []

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, bw, bh = map(float, parts[1:])
                    # денормализуем и переводим в xyxy
                    x_min = (x_center - bw / 2) * w
                    y_min = (y_center - bh / 2) * h
                    x_max = (x_center + bw / 2) * w
                    y_max = (y_center + bh / 2) * h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id + 1)  # +1 потому что 0 — фон у Faster R-CNN

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        target: Dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        # приводим изображение к float32
        image = convert_image_dtype(image, torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    return tuple(zip(*batch))


def list_split_dataset(data_dir: str, split: str) -> List[Tuple[str, str]]:
    """
    Возвращает список пар (путь_к_png, путь_к_txt) для подпапки split.

    Ожидаем структуру:
    dataset/
      train/images/*.png
      train/labels/*.txt
      val/images/*.png
      val/labels/*.txt
      test/images/*.png
      test/labels/*.txt
    """

    img_dir = os.path.join(data_dir, split, "images")
    lbl_dir = os.path.join(data_dir, split, "labels")
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Не найден каталог с изображениями: {img_dir}")
    if not os.path.isdir(lbl_dir):
        raise RuntimeError(f"Не найден каталог с разметкой: {lbl_dir}")

    samples: List[Tuple[str, str]] = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(".png"):
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            txt_path = os.path.join(lbl_dir, base + ".txt")
            samples.append((img_path, txt_path))
    samples.sort()
    if not samples:
        raise RuntimeError(f"В каталоге {img_dir} не найдено изображений .png")
    return samples


def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    epoch_losses = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_losses.append(losses.item())
    return epoch_losses


def evaluate_map(model, data_loader, device):
    if MeanAveragePrecision is None:
        print("TorchMetrics не установлен. Запустите !pip install torchmetrics")
        return None
    metric = MeanAveragePrecision()
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            # подготовка для torchmetrics
            outputs_cpu = []
            for out in outputs:
                outputs_cpu.append(
                    {
                        "boxes": out["boxes"].cpu(),
                        "scores": out["scores"].cpu(),
                        "labels": out["labels"].cpu(),
                    }
                )
            targets_cpu = []
            for t in targets:
                targets_cpu.append({"boxes": t["boxes"], "labels": t["labels"]})
            metric.update(outputs_cpu, targets_cpu)
    return metric.compute()


def plot_curves(history: Dict[str, List[float]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history.get("train_loss", []), label="Train loss")
    if "val_map" in history:
        plt.plot(history["val_map"], label="Val mAP")
    plt.xlabel("Эпоха")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path)
    print(f"График сохранен в {plot_path}")


def run_training(cfg: TrainConfig):
    print("Конфиг:", asdict(cfg))
    train_samples = list_split_dataset(cfg.data_dir, "train")
    val_samples = list_split_dataset(cfg.data_dir, "val")
    test_samples = list_split_dataset(cfg.data_dir, "test")
    print(
        f"Найдено: train={len(train_samples)} файлов, val={len(val_samples)} файлов, test={len(test_samples)} файлов"
    )

    train_ds = YoloTxtDataset(train_samples)
    val_ds = YoloTxtDataset(val_samples)
    test_ds = YoloTxtDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=1 + 1)  # один класс объекта + фон
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_map": []}

    for epoch in range(cfg.num_epochs):
        losses = train_one_epoch(model, optimizer, train_loader, device)
        mean_loss = sum(losses) / len(losses)
        history["train_loss"].append(mean_loss)
        print(f"Эпоха {epoch+1}/{cfg.num_epochs}: средний train loss = {mean_loss:.4f}")

        val_metrics = evaluate_map(model, val_loader, device)
        if val_metrics and "map" in val_metrics:
            history["val_map"].append(val_metrics["map"].item())
            print(f"mAP@0.5: {val_metrics['map']:.4f}")
        else:
            history["val_map"].append(float("nan"))

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, os.path.join(cfg.output_dir, "model.pt"))
    print(f"Модель сохранена в {cfg.output_dir}/model.pt")

    # Финальная оценка на тестовой выборке
    test_metrics = evaluate_map(model, test_loader, device)
    if test_metrics:
        print("Тестовые метрики:", test_metrics)

    plot_curves(history, cfg.output_dir)

    return model, history, test_metrics


if __name__ == "__main__":
    # Этот блок удобно запускать в Colab после монтирования Google Drive.
    # Пример:
    # from google.colab import drive
    # drive.mount('/content/drive')
    # cfg = TrainConfig(data_dir='/content/drive/MyDrive/dataset', num_epochs=5)  # внутри каталоги train/val/test
    # run_training(cfg)
    data_dir = os.environ.get("DATA_DIR", "./data")
    cfg = TrainConfig(data_dir=data_dir)
    run_training(cfg)
