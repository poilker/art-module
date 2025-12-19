import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(arch)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ckpt", type=str, default="outputs/best.pt")
    ap.add_argument("--out", type=str, default="outputs/confusion_matrix.png")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    model = build_model(ckpt["arch"], num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.append(p)
        y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
