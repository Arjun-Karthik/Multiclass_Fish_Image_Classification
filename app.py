import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from PIL import Image
import streamlit as st
import plotly.express as px

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "fish_dataset/train"   # used only to recover class names/order
MODEL_DIR = "Models"               # folder containing model .pkl files

# Utilities
@st.cache_resource(show_spinner=False)
def get_class_names(train_dir: str):
    ds = datasets.ImageFolder(train_dir) 
    return ds.classes

# CNN from scratch definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size=128):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        flatten_size = 128 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_pretrained(model_func, num_classes):
    model = model_func(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze backbone

    if model_func == models.vgg16:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model.to(DEVICE)

    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
        else:
            in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    return model.to(DEVICE)

# Dynamically load models from folder
@st.cache_resource
def load_models_from_folder(model_dir, num_classes):
    loaded_models = {}
    missing_models = []

    if not os.path.exists(model_dir):
        st.warning(f"Model directory `{model_dir}` does not exist.")
        return loaded_models, missing_models

    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            model_name = file.replace("_best.pkl", "")
            path = os.path.join(model_dir, file)
            try:
                if model_name.lower() == "resnet50":
                    model = build_pretrained(models.resnet50, num_classes)
                elif model_name.lower() == "vgg16":
                    model = build_pretrained(models.vgg16, num_classes)
                elif model_name.lower() == "inceptionv3":
                    model = models.inception_v3(pretrained=True)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                elif model_name.lower() == "efficientnetb0":
                    model = build_pretrained(models.efficientnet_b0, num_classes)
                elif model_name.lower() == "mobilenetv2":
                    model = build_pretrained(models.mobilenet_v2, num_classes)
                else:
                    model = SimpleCNN(num_classes)

                state = torch.load(path, map_location=DEVICE)
                model.load_state_dict(state)
                model.to(DEVICE)
                model.eval()
                loaded_models[model_name] = model
            except Exception as e:
                missing_models.append((model_name, str(e)))

    return loaded_models, missing_models

# Per-model transforms (match training!)
@st.cache_resource(show_spinner=False)
def get_transforms():
    def tfm(size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    return {
        "SimpleCNN": tfm(128),
        "VGG16": tfm(224),
        "ResNet50": tfm(224),
        "MobileNetV2": tfm(224),
        "InceptionV3": tfm(299),
        "EfficientNetB0": tfm(224),
    }

def predict_one(model: torch.nn.Module, img: Image.Image, transform, class_names):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)
    top_prob, top_idx = torch.max(probs, dim=0)
    pred_class = class_names[int(top_idx)]
    return pred_class, float(top_prob), probs.tolist()

def show_confidence_bar_chart(results_dict):
    fig = px.bar(
        x=list(results_dict.keys()),
        y=[v * 100 for v in results_dict.values()],
        labels={'x': 'Model', 'y': 'Confidence Score (%)'},
        title="Model Confidence Comparison"
    )
    fig.update_traces(marker_color='skyblue', text=[f"{v*100:.2f}%" for v in results_dict.values()], textposition='outside')
    st.plotly_chart(fig, use_container_width=True, key="confidence_bar_chart")


# UI
st.set_page_config(
    page_title="Fish Classifier", 
    page_icon = "🐟",
    layout="wide"
)

st.markdown("<h1 style = 'text-align : center;'>🐟 Multiclass Fish Image Classification</h>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

with st.sidebar:
    st.header("Model Files")
    st.caption(f"Model folder: `{MODEL_DIR}`")
    st.write("Detected models:")
    for file in os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []:
        if file.endswith(".pkl"):
            st.write(f"- {file}")

# Prepare classes and load models
class_names = get_class_names(TRAIN_DIR)
models_dict, missing_models = load_models_from_folder(MODEL_DIR, num_classes=len(class_names))
tfms = get_transforms()

if missing_models:
    st.warning("Some models failed to load:")
    for name, err in missing_models:
        st.write(f"- **{name}**: {err}")

uploaded = st.file_uploader("Upload a fish image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

col_left, col_right = st.columns([1, 1.2])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col_left.image(img, caption="Uploaded Image", use_container_width=True)

    results_dict = {}
    results_details = []

    with st.spinner("Running predictions across models..."):
        for name, model in models_dict.items():
            pred, conf, probs = predict_one(model, img, tfms.get(name, tfms["SimpleCNN"]), class_names)
            results_details.append({
                "model": name,
                "prediction": pred,
                "confidence": conf,
                "probs": probs,
            })

    # Sort results by confidence descending
    results_details = sorted(results_details, key=lambda x: x["confidence"], reverse=True)
    results_dict = {r["model"]: r["confidence"] for r in results_details}

    col_right.subheader("Model Predictions (High → Low Confidence)")
    summary_rows = [
        f"**{r['model']}** → {r['prediction']}  |  Confidence: {r['confidence']*100:.2f}%"
        for r in results_details
    ]
    col_right.markdown("\n\n".join(summary_rows))

    st.markdown("---")
    st.subheader("Confidence Comparison")
    show_confidence_bar_chart(results_dict)

    st.markdown("---")
    st.subheader("Per-model Confidence Scores")
    for r in results_details:
        with st.expander(f"{r['model']} — {r['prediction']} ({r['confidence']*100:.2f}%)", expanded=False):
            prob_dict = {cls: p for cls, p in zip(class_names, r["probs"])}
            prob_percent = {k: float(v)*100.0 for k, v in prob_dict.items()}
            prob_percent = dict(sorted(prob_percent.items(), key=lambda kv: kv[1], reverse=True))
            st.bar_chart(prob_percent)

    best_model = results_details[0]  # first one after sorting
    st.success(f"**Top Overall**: {best_model['model']} predicts **{best_model['prediction']}** with **{best_model['confidence']*100:.2f}%** confidence")
else:
    st.info("Upload an image to get predictions from all models.")

st.caption(f"Device: **{DEVICE}**")

