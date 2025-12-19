import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import requests

BEST_ID = "1HSyvm7HIOWIj1E5cXjuiRBTKNd4-ZSIQ"
# LAST_ID = "1dtnQgSbuqdudPfbPYDEA_H3BQX3djyFx"  # 需要才用

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
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)

@st.cache_resource
def ensure_best_model() -> Path:
    best_path = Path("outputs/best.pt")
    if not best_path.exists():
        st.info("Downloading model (best.pt) from Google Drive...")
        download_from_gdrive(BEST_ID, best_path)
        st.success("Model downloaded ✅")
    return best_path

def torch_load_compat(path: Path):
    """
    PyTorch >=2.6 預設 weights_only=True 可能讀不了你這種 checkpoint(dict)。
    我們優先嘗試一般 torch.load，失敗再用 weights_only=False。
    """
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e1:
        # torch>=2.6 才支援 weights_only 參數；舊版會 TypeError
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # 舊版 PyTorch 沒這參數，回退一般 load
            return torch.load(path, map_location="cpu")

# === Paths ===
CKPT_PATH = Path("outputs/best.pt")
CM_PATH = Path("outputs/confusion_matrix.png")

st.set_page_config(page_title="Style Classifier", page_icon="🎨", layout="centered")
st.title("🎨 Painting Style Classifier (5 classes)")

# Confusion matrix (optional)
if CM_PATH.exists():
    st.subheader("Confusion Matrix (Test)")
    st.image(str(CM_PATH), use_container_width=True)
else:
    st.info("找不到 outputs/confusion_matrix.png（可先跑 eval.py 產生它）")

@st.cache_resource
def load_ckpt_and_model():
    # 先確保 best.pt 存在（沒有就下載）
    ensure_best_model()

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔：{CKPT_PATH.resolve()}")

    ckpt = torch_load_compat(CKPT_PATH)

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
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, tf, class_names

def predict_topk(model, tf, class_names, img: Image.Image, k=3):
    x = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    k = min(k, len(class_names))
    top = torch.topk(probs, k=k)
    return [(class_names[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]

try:
    model, tf, class_names = load_ckpt_and_model()
    st.success(f"Model loaded ✅ classes={class_names}")
except Exception as e:
    st.error(f"模型載入失敗：{e}")
    st.stop()

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

