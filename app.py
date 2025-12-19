import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# === Config ===
CKPT_PATH = Path("outputs/best.pt")
CM_PATH = Path("outputs/confusion_matrix.png")

st.set_page_config(page_title="Style Classifier", page_icon="🎨", layout="centered")
st.title("🎨 Painting Style Classifier (5 classes)")

# Show confusion matrix image if exists (no matplotlib needed)
if CM_PATH.exists():
    st.subheader("Confusion Matrix (Test)")
    st.image(str(CM_PATH), use_container_width=True)
else:
    st.info("找不到 outputs/confusion_matrix.png（你可以先跑 eval.py 產生它）")

@st.cache_resource
def load_ckpt_and_model():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔：{CKPT_PATH.resolve()}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # 支援兩種常見格式：
    # A) 你 train.py 存的是 {'state_dict':..., 'class_names':..., 'arch':...}
    # B) 或者只存 model.state_dict()
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        class_names = ckpt.get("class_names", None)
        arch = ckpt.get("arch", "resnet18")
        state_dict = ckpt["state_dict"]
    else:
        class_names = None
        arch = "resnet18"
        state_dict = ckpt

    # 如果沒 class_names，就用你資料夾名稱順序（跟 train.py 印出的一致）
    if class_names is None:
        class_names = ["Baroque", "Japanese_Art", "Realism", "Renaissance", "Romanticism"]

    # Build model
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

# Load model once
try:
    model, tf, class_names = load_ckpt_and_model()
    st.success(f"Model loaded ✅ classes={class_names}")
except Exception as e:
    st.error(str(e))
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
