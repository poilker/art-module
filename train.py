import argparse, json, os, random, time
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def plot_curves(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        p = torch.argmax(logits, dim=1)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(p.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_backbone", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = get_device()
    print("Device:", device)

    # ImageNet normalize（pretrained backbone 對齊）
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.15)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=eval_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    model = build_model(args.arch, num_classes=num_classes, pretrained=True)

    if args.freeze_backbone:
        # freeze all except last classifier layer
        for name, p in model.named_parameters():
            p.requires_grad = False
        # unfreeze head
        if args.arch == "resnet18":
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            for p in model.classifier[-1].parameters():
                p.requires_grad = True
        print("Backbone frozen; training head only.")

    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    metrics = {"classes": class_names, "arch": args.arch, "device": str(device), "best_val_macro_f1": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        all_y, all_p = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * x.size(0)
            p = torch.argmax(logits, dim=1)
            all_y.append(y.detach().cpu().numpy())
            all_p.append(p.detach().cpu().numpy())

            pbar.set_postfix(loss=loss.item())

        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_p)
        train_loss = epoch_loss / len(train_ds)
        train_acc = accuracy_score(y_true, y_pred)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}")

        # save last
        last_path = os.path.join(args.out_dir, "last.pt")
        torch.save({
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "img_size": args.img_size,
        }, last_path)

        # save best by macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save({
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": args.img_size,
            }, best_path)

    metrics["best_val_macro_f1"] = float(best_f1)

    # final test
    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model = build_model(ckpt["arch"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
    metrics["test_loss"] = float(test_loss)
    metrics["test_acc"] = float(test_acc)
    metrics["test_macro_f1"] = float(test_f1)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_curves(history, os.path.join(args.out_dir, "training_curves.png"))
    print("Done. Metrics saved to outputs/metrics.json")


if __name__ == "__main__":
    main()
