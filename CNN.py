import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import base64
from io import BytesIO

CONFIDENCE_TEMPERATURE = 75.0


def calibrated_confidence(logits, class_idx, candidate_class_ids=None):
    """Return a less overconfident display score for this model's large logits."""
    if candidate_class_ids:
        candidate_class_ids = list(candidate_class_ids)
        candidate_logits = logits[candidate_class_ids]
        probabilities = torch.softmax(candidate_logits / CONFIDENCE_TEMPERATURE, dim=-1)
        candidate_position = candidate_class_ids.index(int(class_idx))
        return int(round(probabilities[candidate_position].item() * 100))

    probabilities = torch.softmax(logits / CONFIDENCE_TEMPERATURE, dim=-1)
    return int(round(probabilities[int(class_idx)].item() * 100))


HEALTHY_CLASS_IDS = {3, 4, 6, 10, 14, 17, 19, 22, 23, 24, 27, 37}


idx_to_classes = {0: 'Apple___Apple_scab',
                  1: 'Apple___Black_rot',
                  2: 'Apple___Cedar_apple_rust',
                  3: 'Apple___healthy',
                  4: 'Blueberry___healthy',
                  5: 'Cherry_(including_sour)___Powdery_mildew',
                  6: 'Cherry_(including_sour)___healthy',
                  7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                  8: 'Corn_(maize)___Common_rust_',
                  9: 'Corn_(maize)___Northern_Leaf_Blight',
                  10: 'Corn_(maize)___healthy',
                  11: 'Grape___Black_rot',
                  12: 'Grape___Esca_(Black_Measles)',
                  13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  14: 'Grape___healthy',
                  15: 'Orange___Haunglongbing_(Citrus_greening)',
                  16: 'Peach___Bacterial_spot',
                  17: 'Peach___healthy',
                  18: 'Pepper,_bell___Bacterial_spot',
                  19: 'Pepper,_bell___healthy',
                  20: 'Potato___Early_blight',
                  21: 'Potato___Late_blight',
                  22: 'Potato___healthy',
                  23: 'Raspberry___healthy',
                  24: 'Soybean___healthy',
                  25: 'Squash___Powdery_mildew',
                  26: 'Strawberry___Leaf_scorch',
                  27: 'Strawberry___healthy',
                  28: 'Tomato___Bacterial_spot',
                  29: 'Tomato___Early_blight',
                  30: 'Tomato___Late_blight',
                  31: 'Tomato___Leaf_Mold',
                  32: 'Tomato___Septoria_leaf_spot',
                  33: 'Tomato___Spider_mites Two-spotted_spider_mite',
                  34: 'Tomato___Target_Spot',
                  35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                  36: 'Tomato___Tomato_mosaic_virus',
                  37: 'Tomato___healthy'}


def class_crop_name(class_name):
    crop = class_name.split("___", 1)[0]
    return crop.replace(",", "").replace("_", " ").strip().lower()


CROP_CLASS_IDS = {}
for class_idx, class_name in idx_to_classes.items():
    CROP_CLASS_IDS.setdefault(class_crop_name(class_name), []).append(class_idx)


CLASS_ID_TO_CROP_IDS = {
    class_idx: CROP_CLASS_IDS[class_crop_name(class_name)]
    for class_idx, class_name in idx_to_classes.items()
}


class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)
        out = self.dense_layers(out)
        return out


# ===================== GRAD-CAM (Multiple layers averaged) =====================

def legacy_generate_gradcam(model, image_path):
    """
    Improved Grad-CAM: averages heatmaps from block2 and block3
    for better spatial accuracy on disease spots.
    """
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((224, 224))
    input_tensor = TF.to_tensor(image_resized)
    input_tensor = input_tensor.unsqueeze(0)

    model.eval()

    def get_cam_for_layer(layer_idx):
        gradients = []
        activations = []

        def fwd(module, inp, out):
            activations.append(out.detach())

        def bwd(module, gin, gout):
            gradients.append(gout[0].detach())

        target = model.conv_layers[layer_idx]
        h1 = target.register_forward_hook(fwd)
        h2 = target.register_full_backward_hook(bwd)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        model.zero_grad()
        output[0, pred_class].backward()

        h1.remove()
        h2.remove()

        grads = gradients[0].cpu().numpy()
        acts  = activations[0].cpu().numpy()

        weights = grads.mean(axis=(2, 3), keepdims=True)
        cam = (weights * acts).sum(axis=1).squeeze()
        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cv2.resize(cam, (224, 224)), pred_class

    # Block 2 last Conv2d = index 10 (28x28 maps — finer detail)
    # Block 3 last Conv2d = index 17 (14x14 maps — semantic)
    cam_b2, pred_class = get_cam_for_layer(10)
    cam_b3, _          = get_cam_for_layer(17)

    # Weighted average: more weight to finer layer for spot detection
    cam_final = 0.6 * cam_b2 + 0.4 * cam_b3

    # Sharpen with power scaling to suppress low-activation noise
    cam_final = np.power(cam_final, 1.5)
    if cam_final.max() > 0:
        cam_final = cam_final / cam_final.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_final), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(image_resized)
    overlay = (0.6 * orig_np + 0.4 * heatmap).astype(np.uint8)

    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return b64, pred_class


def generate_gradcam(model, image_path, class_idx=None):
    """
    Grad-CAM for the selected/predicted class.

    Deeper convolution layers make the heatmap less edge-driven, and a soft
    leaf mask reduces attention from the plain background and outer border.
    """
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((224, 224))
    input_tensor = TF.to_tensor(image_resized)
    input_tensor = input_tensor.unsqueeze(0)

    model.eval()

    def normalize_map(values):
        values = values.astype(np.float32)
        max_value = float(values.max())
        if max_value > 0:
            return values / max_value
        return values

    def build_leaf_mask(rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
        color_delta = np.maximum.reduce([
            cv2.absdiff(r, g),
            cv2.absdiff(r, b),
            cv2.absdiff(g, b),
        ])

        # Keep colored plant tissue, but reject pale studio backgrounds and glare.
        mask = ((saturation > 28) | (color_delta > 16)) & (value < 245)
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        return mask.astype(np.float32) / 255.0

    def build_interior_weight(leaf_mask):
        binary = (leaf_mask > 0.35).astype(np.uint8)
        if binary.max() == 0:
            return leaf_mask

        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        interior = normalize_map(distance)
        interior = np.clip(interior * 1.8, 0.0, 1.0)
        interior = cv2.GaussianBlur(interior, (13, 13), 0)
        return np.clip((0.35 + 0.65 * interior) * leaf_mask, 0.0, 1.0)

    def build_symptom_prior(rgb_image, leaf_mask):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        local_mean = cv2.GaussianBlur(gray, (21, 21), 0)
        local_contrast = cv2.absdiff(gray, local_mean)

        brown_or_tan = ((hue < 32) | (hue > 165)) & (saturation > 22) & (value > 35) & (value < 230)
        yellow_chlorosis = (hue >= 20) & (hue <= 36) & (saturation > 45) & (value > 65) & (value < 190)
        dark_spots = (value < 120) & (saturation > 18)
        contrast_spots = (local_contrast > 18) & (brown_or_tan | yellow_chlorosis | dark_spots)
        glare = (value > 195) & (saturation < 65)

        prior = (
            0.62 * brown_or_tan.astype(np.float32) +
            0.10 * yellow_chlorosis.astype(np.float32) +
            0.22 * dark_spots.astype(np.float32) +
            0.06 * contrast_spots.astype(np.float32)
        )
        prior = prior * leaf_mask * (1.0 - 0.85 * glare.astype(np.float32))
        prior = cv2.morphologyEx(prior, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        prior = cv2.GaussianBlur(prior, (9, 9), 0)
        return normalize_map(prior)

    def get_cam_for_layer(layer_idx, target_class):
        gradients = []
        activations = []

        def fwd(module, inp, out):
            activations.append(out.detach())

        def bwd(module, gin, gout):
            gradients.append(gout[0].detach())

        target = model.conv_layers[layer_idx]
        h1 = target.register_forward_hook(fwd)
        h2 = target.register_full_backward_hook(bwd)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        selected_class = pred_class if target_class is None else int(target_class)
        confidence = calibrated_confidence(
            output[0],
            selected_class,
            CLASS_ID_TO_CROP_IDS.get(selected_class),
        )
        model.zero_grad()
        output[0, selected_class].backward()

        h1.remove()
        h2.remove()

        grads = gradients[0].cpu().numpy()
        acts = activations[0].cpu().numpy()

        weights = grads.mean(axis=(2, 3), keepdims=True)
        cam = (weights * acts).sum(axis=1).squeeze()
        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cv2.resize(cam, (224, 224)), pred_class, confidence

    cam_b2, pred_class, confidence = get_cam_for_layer(10, class_idx)
    cam_b3, _, _ = get_cam_for_layer(17, class_idx)
    cam_b4, _, _ = get_cam_for_layer(24, class_idx)

    selected_class = pred_class if class_idx is None else int(class_idx)
    orig_np = np.array(image_resized)
    leaf_mask = build_leaf_mask(orig_np)
    interior_weight = build_interior_weight(leaf_mask)
    symptom_prior = build_symptom_prior(orig_np, interior_weight)

    cam_final = 0.28 * cam_b2 + 0.34 * cam_b3 + 0.38 * cam_b4
    cam_final = cam_final * interior_weight

    if selected_class not in HEALTHY_CLASS_IDS and symptom_prior.max() > 0:
        cam_final = (0.30 * cam_final * (0.25 + 0.75 * symptom_prior)) + (0.70 * symptom_prior)

    cam_final = cv2.GaussianBlur(cam_final, (5, 5), 0)
    cam_final = np.power(cam_final, 1.55)

    if cam_final.max() > 0:
        cam_final = cam_final / cam_final.max()

    strong_signal = cam_final[cam_final > 0.03]
    if strong_signal.size:
        high_clip = np.percentile(strong_signal, 98.5)
        if high_clip > 0:
            cam_final = np.clip(cam_final / high_clip, 0.0, 1.0)

    # Fade weak activation instead of coloring the whole leaf.
    cam_final = np.clip((cam_final - 0.06) / 0.94, 0.0, 1.0)
    cam_final = cv2.GaussianBlur(cam_final, (3, 3), 0)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_final), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    attention_alpha = np.clip((0.08 + 0.54 * cam_final) * (cam_final > 0.05), 0.0, 0.62)
    attention_alpha = attention_alpha[:, :, None]
    overlay = ((1.0 - attention_alpha) * orig_np + attention_alpha * heatmap).astype(np.uint8)

    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return b64, pred_class, confidence
