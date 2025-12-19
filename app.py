import streamlit as st
import requests
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================
# Google Drive download
# =========================
BEST_ID = "1HSyvm7HIOWIj1E5cXjuiRBTKNd4-ZSIQ"

def _get_confirm_token(resp: requests.Response):
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    return None

def download_from_gdrive(file_id: str, dest: Path, chunk_size: int = 1024 * 1024):
    url = "https://docs.google.com/uc?export=download"
    sess = requests.Session()

    resp = sess.get(url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(resp)
    if token:
        resp = sess.get(url, params={"id": file_id, "confirm": token}, stream=True)

    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0

    prog = st.progress(0) if total > 0 else None

    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if prog and total > 0:
                prog.progress(min(downloaded / total, 1.0))

    tmp.replace(dest)

@st.cache_resource
def ensure_best_model() -> str:
    """確保 outputs/best.pt 存在；不存在就從 GDrive 下載。"""
    best_path = Path("outputs/best.pt")
    if not best_path.exists() or best_path.stat().st_size == 0:
        st.info("Downloading model (best.pt) from Google Drive...")
        download_from_gdrive(BEST_ID, best_path)
        st.success("Model downloaded ✅")
    return str(best_path)

# =========================
# UI Config
# =========================
st.set_page_config(page_title="Style Classifier", page_icon="🎨", layout="centered")
st.title("🎨 Painting Style Classifier (5 classes)")

# paths
CKPT_PATH = Path(ensure_best_model())          # ✅ 這行是關鍵：雲端會先下載再回傳路徑
CM_PATH = Path("outputs/confusion_matrix.png")

# Show confusion matrix image if exists
if CM_PATH.exists():
    st.subheader("Confusion Matrix (Test)")
    st.image(str(CM_PATH), use_container_width=True)
else:
    st.info("找不到 outputs/confusion_matrix.png（你可以先跑 eval.py 產生它，或不放也沒關係）")

# =========================
# Load model
# =========================
@st.cache_resource
def load_ckpt_and_model(ckpt_path_str: str):
    ckpt_path = Path(ckpt_path_str)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 支援兩種格式：
    # A) {'state_dict':..., 'class_names':..., 'arch':...}
    # B) 純 state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        class_names = ckpt.get("class_names", None)
        arch = ckpt.get("arch", "resnet18")
        state_dict = ckpt["state_dict"]
    else:
        class_names = None
        arch = "resnet18"
        state_dict = ckpt

    if class_names is None:
        class_names = ["Baroque", "Japanese_Art", "Realism", "Renaissance", "Romanticism"]

    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, tf, class_names

def predict_topk(model, tf, class_names, img: Image.Image, k=3):
    x = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    k = min(k, len(class_names))
    top = torch.topk(probs, k=k)
    return [(class_names[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]

# Load model once
try:
    model, tf, class_names = load_ckpt_and_model(str(CKPT_PATH))
    st.success(f"Model loaded ✅ classes={class_names}")
except Exception as e:
    st.error(f"模型載入失敗：{e}")
    st.stop()

# =========================
# Inference UI
# =========================
st.subheader("Try your own image")
uploaded = st.file_uploader("Upload an image (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Input", use_container_width=True)

    results = predict_topk(model, tf, class_names, img, k=3)
    st.markdown("### Top-3 predictions")
    for name, p in results:
        st.write(f"- **{name}**: {p*100:.2f}%")
else:
    st.info("請先上傳一張圖片。")


