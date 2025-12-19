import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import requests

# ===== Google Drive file id =====
BEST_ID = "1HSyvm7HIOWIj1E5cXjuiRBTKNd4-ZSIQ"
# LAST_ID = "1dtnQgSbuqdudPfbPYDEA_H3BQX3djyFx"  # 需要才用

CKPT_PATH = Path("outputs/best.pt")
CM_PATH = Path("outputs/confusion_matrix.png")

st.set_page_config(page_title="Style Classifier", page_icon="🎨", layout="centered")
st.title("🎨 Painting Style Classifier (5 classes)")

def _get_confirm_token(resp: requests.Response):
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    return None

def _looks_like_html(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(256).lstrip()
        return head.startswith(b"<")  # <html / <!DOCTYPE ...
    except Exception:
        return False

def download_from_gdrive(file_id: str, dest: Path, chunk_size: int = 1024 * 1024):
    """
    Download a file from Google Drive public link.
    If it downloads an HTML page (permission/quota), we'll detect later.
    """
    url = "https://docs.google.com/uc?export=download"
    sess = requests.Session()

    resp = sess.get(url, params={"id": file_id}, stream=True, timeout=60)
    token = _get_confirm_token(resp)
    if token:
        resp = sess.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)

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
    """
    Ensure outputs/best.pt exists.
    If it's HTML (bad download), delete and re-download.
    """
    if CKPT_PATH.exists() and _looks_like_html(CKPT_PATH):
        CKPT_PATH.unlink(missing_ok=True)

    if not CKPT_PATH.exists():
        st.info("Downloading model (best.pt) from Google Drive...")
        download_from_gdrive(BEST_ID, CKPT_PATH)

        # verify
        if _looks_like_html(CKPT_PATH):
            CKPT_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                "下載到的 best.pt 看起來是 HTML（不是模型檔）。\n"
                "請確認 Google Drive 權限是「任何知道連結的人都可檢視」，且沒有流量/防毒確認頁。\n"
                "（最穩的替代方案：改放 GitHub Releases / HuggingFace）"
            )
        st.success("Model downloaded ✅")
    return CKPT_PATH

def build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")

@st.cache_resource
def load_ckpt_and_model():
    ckpt_path = ensure_best_model()

    # PyTorch 2.6+ default weights_only=True may break older checkpoints
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        class_names = ckpt.get("class_names") or ["Baroque", "Japanese_Art", "Realism", "Renaissance", "Romanticism"]
        arch = ckpt.get("arch", "resnet18")
        state_dict = ckpt["state_dict"]
    else:
        # fallback: pure state_dict
        class_names = ["Baroque", "Japanese_Art", "Realism", "Renaissance", "Romanticism"]
        arch = "resnet18"
        state_dict = ckpt

    model = build_model(arch, num_classes=len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tf, class_names

def predict_topk(model, tf, class_names, img: Image.Image, k=3):
    x = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    k = min(k, len(class_names))
    top = torch.topk(probs, k=k)
    return [(class_names[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]

# ===== UI =====
if CM_PATH.exists():
    st.subheader("Confusion Matrix (Test)")
    st.image(str(CM_PATH), use_container_width=True)
else:
    st.info("找不到 outputs/confusion_matrix.png（可先跑 eval.py 產生）")

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
