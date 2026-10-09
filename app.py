import io
from pathlib import Path
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import os
import json
import uuid
import gdown
from groq import Groq
from flask import send_file
from werkzeug.utils import secure_filename
from pdf_report import generate_pdf_report
from database import (init_db, save_detection, get_all_detections,
                      get_disease_counts, get_daily_counts, get_stats, clear_history)

# ========== Google Drive Auto Model Download ==========
model_path = "plant_disease_model_1_latest.pt"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    try:
        gdown.download("https://drive.google.com/uc?id=1XBo4fdRs3mihkqfIYhgnfr5a79b63ZUo", model_path, quiet=False)
    except Exception as e:
        print("Model download failed:", e)
        import sys
        sys.exit(1)
# =======================================================

# ========== Groq Client Setup ==========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
BASE_DIR = Path(__file__).resolve().parent

disease_info = pd.read_csv(BASE_DIR / "disease_info.csv", encoding='cp1252')

MODEL_CLASS_TO_DISEASE_NAME = {
    'Apple___Apple_scab': 'Apple : Scab',
    'Apple___Black_rot': 'Apple : Black Rot',
    'Apple___Cedar_apple_rust': 'Apple : Cedar rust',
    'Apple___healthy': 'Apple : Healthy',
    'Blueberry___healthy': 'Blueberry : Healthy',
    'Cherry_(including_sour)___healthy': 'Cherry : Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry : Powdery Mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Corn : Cercospora Leaf Spot | Gray Leaf Spot',
    'Corn_(maize)___Common_rust_': 'Corn : Common Rust',
    'Corn_(maize)___healthy': 'Corn : Healthy',
    'Corn_(maize)___Northern_Leaf_Blight': 'Corn : Northern Leaf Blight',
    'Grape___Black_rot': 'Grape : Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape : Esca | Black Measles',
    'Grape___healthy': 'Grape : Healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape : Leaf Blight | Isariopsis Leaf Spot',
    'Orange___Haunglongbing_(Citrus_greening)': 'Orange : Haunglongbing | Citrus Greening',
    'Peach___Bacterial_spot': 'Peach : Bacterial Spot',
    'Peach___healthy': 'Peach : Healthy',
    'Pepper,_bell___Bacterial_spot': 'Pepper bell : Bacterial Spot',
    'Pepper,_bell___healthy': 'Pepper bell : Healthy',
    'Potato___Early_blight': 'Potato : Early Blight',
    'Potato___healthy': 'Potato : Healthy',
    'Potato___Late_blight': 'Potato : Late Blight',
    'Raspberry___healthy': 'Raspberry : Healthy',
    'Soybean___healthy': 'Soybean : Healthy',
    'Squash___Powdery_mildew': 'Squash : Powdery Mildew',
    'Strawberry___healthy': 'Strawberry : Healthy',
    'Strawberry___Leaf_scorch': 'Strawberry : Leaf Scorch',
    'Tomato___Bacterial_spot': 'Tomato : Bacterial Spot',
    'Tomato___Early_blight': 'Tomato : Early Blight',
    'Tomato___healthy': 'Tomato : Healthy',
    'Tomato___Late_blight': 'Tomato : Late Blight',
    'Tomato___Leaf_Mold': 'Tomato : Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato : Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato : Spider Mites | Two-Spotted Spider Mite',
    'Tomato___Target_Spot': 'Tomato : Target Spot',
    'Tomato___Tomato_mosaic_virus': 'Tomato : Mosaic Virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato : Yellow Leaf Curl Virus',
}

model = CNN.CNN(38)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)
init_db()


def get_disease_record(pred_index: int):
    model_class = CNN.idx_to_classes[int(pred_index)]
    disease_name = MODEL_CLASS_TO_DISEASE_NAME[model_class]
    match = disease_info[disease_info["disease_name"] == disease_name]
    if match.empty:
        raise KeyError(f"No disease_info.csv row for model class {model_class}")
    return match.iloc[0]


def first_sentences(text, limit=2):
    clean = " ".join(str(text).replace("\n", " ").split())
    if not clean:
        return ""
    parts = [p.strip() for p in clean.split(". ") if p.strip()]
    selected = ". ".join(parts[:limit])
    return selected if selected.endswith(".") else selected + "."


def plant_name_from_disease(disease_name: str) -> str:
    return str(disease_name).split(":", 1)[0].strip()


def is_healthy_prediction(disease_name: str) -> bool:
    return "healthy" in str(disease_name).lower()


def build_local_xai(disease_name, disease, model_confidence):
    plant_name = plant_name_from_disease(disease_name)

    if disease is not None:
        overview = first_sentences(disease["description"], 2)
        action = first_sentences(disease["Possible Steps"], 2)
    else:
        overview = (
            f"The model predicted {disease_name}. Compare the highlighted area "
            "with the original image and confirm the symptoms manually."
        )
        action = (
            "Inspect the plant closely, isolate affected leaves if needed, "
            "and consult a local plant expert before applying treatment."
        )

    if is_healthy_prediction(disease_name):
        overview = (
            f"The model classified this {plant_name.lower()} leaf as healthy. "
            "No clear disease pattern was detected in the uploaded image."
        )
        action = (
            "Continue routine plant care: keep watering consistent, provide good airflow, "
            "avoid wetting leaves unnecessarily, and monitor for new spots or discoloration."
        )
        heatmap = (
            "For a healthy prediction, the Grad-CAM overlay shows normal leaf regions that supported "
            "the model's decision. Any highlighted area should be treated as model attention, not as a disease symptom."
        )
        focus = (
            f"For {disease_name}, the model is likely responding to normal color, texture, venation, "
            "and leaf shape patterns learned during training."
        )
    else:
        heatmap = (
            "The Grad-CAM overlay marks image regions that most influenced this CNN prediction. "
            "Use it as an attention guide, then compare the highlighted area with visible spots, "
            "discoloration, mold, or damaged tissue in the original leaf."
        )
        focus = (
            f"For {disease_name}, the model is likely reacting to learned color, texture, "
            "and lesion patterns from training images. Grad-CAM is not an exact symptom boundary, "
            "so highlighted borders or background areas should be treated as weak evidence."
        )

    return {
        "overview": overview,
        "heatmap": heatmap,
        "focus": focus,
        "action": action,
        "confidence": model_confidence,
    }


def normalize_crop_name(crop_name):
    if not crop_name:
        return ""
    crop_key = str(crop_name).replace(",", "").replace("_", " ").strip().lower()
    crop_aliases = {
        "cherry": "cherry (including sour)",
        "corn": "corn (maize)",
        "corn maize": "corn (maize)",
        "pepper": "pepper bell",
    }
    return crop_aliases.get(crop_key, crop_key)


def prediction(image_path, crop_name=None):
    result = predict_image(image_path, crop_name)
    return result["pred"], result["confidence"]


def predict_image(image_path, crop_name=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)
    crop_key = normalize_crop_name(crop_name)
    allowed_class_ids = CNN.CROP_CLASS_IDS.get(crop_key)
    if allowed_class_ids:
        candidate_ids = list(allowed_class_ids)
    else:
        candidate_ids = list(range(output.shape[1]))

    candidate_logits = output[0, candidate_ids]
    candidate_probs = torch.softmax(candidate_logits / CNN.CONFIDENCE_TEMPERATURE, dim=-1)
    top_count = min(3, len(candidate_ids))
    top_probs, top_positions = torch.topk(candidate_probs, top_count)

    index = int(candidate_ids[int(top_positions[0].item())])
    confidence = int(round(top_probs[0].item() * 100))
    alternatives = []
    for probability, position in zip(top_probs, top_positions):
        class_id = int(candidate_ids[int(position.item())])
        alternatives.append({
            "pred": class_id,
            "name": get_disease_record(class_id)["disease_name"],
            "confidence": int(round(probability.item() * 100)),
        })

    return {
        "pred": index,
        "confidence": confidence,
        "alternatives": alternatives,
        "crop_constrained": bool(allowed_class_ids),
    }


def unique_upload_filename(filename):
    safe_name = secure_filename(filename or "upload.jpg")
    stem = Path(safe_name).stem or "upload"
    suffix = Path(safe_name).suffix.lower() or ".jpg"
    return f"{uuid.uuid4().hex}_{stem}{suffix}"


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/contact')
def contact():
    return render_template('contact-us.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = unique_upload_filename(image.filename)

        upload_folder = BASE_DIR / "static"
        upload_folder.mkdir(exist_ok=True)
        file_path = upload_folder / filename
        image.save(file_path)

        selected_crop = request.form.get('crop', '')
        prediction_result = predict_image(file_path, selected_crop)
        pred = prediction_result["pred"]
        confidence = prediction_result["confidence"]
        disease = get_disease_record(pred)
        title = disease['disease_name']
        description = disease['description']
        prevent = disease['Possible Steps']
        save_detection(title, filename, pred, confidence)

        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               uploaded_image_url=f"/static/{filename}",
                               reference_image_url=disease['image_url'],
                               pred=pred, filename=filename,
                               confidence=confidence,
                               alternatives=prediction_result["alternatives"],
                               crop_constrained=prediction_result["crop_constrained"])


@app.route('/gradcam/<filename>/<int:pred>')
def gradcam(filename, pred):
    file_path = BASE_DIR / "static" / filename

    if not file_path.exists():
        return redirect('/index')

    try:
        gradcam_b64, _, confidence = CNN.generate_gradcam(model, file_path, class_idx=pred)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return redirect('/index')

    disease = get_disease_record(pred)
    title = disease['disease_name']

    return render_template('gradcam.html',
                       gradcam_image=gradcam_b64,
                       title=title,
                       pred=pred,
                       original_image=f"/static/{filename}",
                       filename=filename,
                       confidence=confidence)   


# ========== Groq XAI Explanation Route =================
@app.route('/api/xai-explain', methods=['POST'])
def xai_explain():
    import traceback
    try:
        # ── 1. Parse request ──
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid request body"}), 400

        disease_name = data.get('disease', 'Unknown Disease')
        model_confidence = data.get('confidence')
        try:
            model_confidence = int(model_confidence)
        except (TypeError, ValueError):
            model_confidence = None
        print(f"[XAI] Request received for: {disease_name}")

        match = disease_info[disease_info["disease_name"] == disease_name]
        disease = match.iloc[0] if not match.empty else None
        result = build_local_xai(disease_name, disease, model_confidence)

        print(f"[XAI] Success! Confidence: {result['confidence']}%")
        return jsonify(result)

        # ── 2. Build Groq client fresh each call (avoids stale client bug) ──
        groq_client = Groq(api_key=GROQ_API_KEY)

        # ── 3. Prompt — NO angle brackets inside JSON, clean format ──
        prompt = (
            f'You are a plant pathology and Explainable AI expert.\n'
            f'A CNN model detected "{disease_name}" from a leaf image using Grad-CAM.\n\n'
            f'Reply with ONLY this JSON, no markdown, no extra text:\n'
            f'{{\n'
            f'  "overview": "What is {disease_name}? Which plants? How serious? (2-3 sentences)",\n'
            f'  "heatmap": "Explain that the Grad-CAM overlay shows high-attention CNN regions, not guaranteed exact symptoms. Describe likely visual cues for {disease_name}; do not claim the model focused on edges unless the image clearly supports it. (2-3 sentences)",\n'
            f'  "focus": "Why might those disease regions influence the model? Mention uncertainty and avoid overclaiming. (2 sentences)",\n'
            f'  "action": "What should a farmer do immediately? (2-3 sentences)"\n'
            f'}}\n\n'
            f'Replace all other values with real content about {disease_name}.\n'
            f'Do not estimate or mention confidence; the application will use CNN softmax confidence separately.\n'
            f'Output ONLY the JSON object.'
        )

        # ── 4. Call Groq ──
        print("[XAI] Calling Groq API...")
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )

        raw = response.choices[0].message.content.strip()
        print(f"[XAI] Raw response: {raw[:200]}")

        # ── 5. Clean & parse JSON ──
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Find JSON object in response
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in response: {raw[:100]}")

        raw = raw[start:end]
        result = json.loads(raw)

        # ── 6. Validate keys ──
        required = {"overview", "heatmap", "focus", "action"}
        missing  = required - set(result.keys())
        if missing:
            raise ValueError(f"Missing keys in response: {missing}")

        # ── 7. Ensure confidence is int ──
        result['confidence'] = model_confidence

        print(f"[XAI] Success! Confidence: {result['confidence']}%")
        return jsonify(result)

    except json.JSONDecodeError as e:
        print(f"[XAI] JSON parse error: {e}\nRaw: {raw}")
        return jsonify({"error": f"AI format error: {str(e)}"}), 500
    except Exception as e:
        print(f"[XAI] FULL ERROR:\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
# =======================================================


@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/hybridaction/'):
        return '', 204
    return render_template('404.html'), 404


# ═══════════════════════════════════════════════════════════════════════════
#  HISTORY DASHBOARD ROUTES — app.py mein add karo
#
#  Step 1: Top pe yeh imports add karo:
#     from database import init_db, save_detection, get_all_detections, get_disease_counts, get_daily_counts, get_stats, clear_history
#
#  Step 2: app = Flask(__name__) ke BAAD yeh line add karo:
#     init_db()
#
#  Step 3: /submit route mein pred return se PEHLE yeh line add karo:
#     save_detection(title, filename, pred)
#
#  Step 4: Neeche wale dono routes apne existing routes ke saath paste karo
# ═══════════════════════════════════════════════════════════════════════════



# ── /submit route mein, return render_template se PEHLE add karo ──
# save_detection(title, filename, pred)


# ── Yeh dono nayi routes paste karo ──

@app.route('/history')
def history_dashboard():
    detections  = get_all_detections(100)
    disease_counts = get_disease_counts()
    daily_counts   = get_daily_counts(14)
    stats          = get_stats()
    return render_template('history.html',
                           detections=detections,
                           disease_counts=disease_counts,
                           daily_counts=daily_counts,
                           stats=stats)


@app.route('/history/clear', methods=['POST'])
def clear_history_route():
    clear_history()
    return redirect('/history')

@app.route('/evaluation')
def evaluation():
    eval_results_path = BASE_DIR / "static" / "eval_results.json"
    eval_results = {}
    if eval_results_path.exists():
        try:
            eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Could not load evaluation results: {e}")
    return render_template('evaluation.html', eval_results=eval_results)
# ═══════════════════════════════════════════════════════════════════════════
#  PDF REPORT ROUTE — 


@app.route('/download-report/<filename>/<int:pred>')
def download_report(filename, pred):
    """
    PDF report download karo disease detection result ka.
    URL format: /download-report/<image_filename>/<pred_index>
    """
    file_path = BASE_DIR / "static" / filename

    if not file_path.exists():
        return redirect('/index')

    # Grad-CAM generate karo
    confidence = None
    try:
        gradcam_b64, _, confidence = CNN.generate_gradcam(model, file_path, class_idx=pred)
    except Exception as e:
        print(f"Grad-CAM error in report: {e}")
        gradcam_b64 = ""

    disease = get_disease_record(pred)
    title       = disease['disease_name']
    description = disease['description']
    prevent     = disease['Possible Steps']

    # XAI explanation — agar Groq se already mila hua ho toh pass karo
    # Warna None pass karo aur woh section skip ho jayega
    xai_explanation = None
    # Example agar Groq call karna ho yahan bhi:
    # xai_explanation = {
    #     "overview": "...",
    #     "heatmap_focus": "...",
    #     "model_reasoning": "...",
    #     "recommended_action": "..."
    # }

    # PDF bytes generate karo
    try:
        pdf_bytes = generate_pdf_report(
            title=title,
            description=description,
            prevent=prevent,
            original_image_path=file_path,
            gradcam_b64=gradcam_b64,
            confidence=confidence,
            xai_explanation=xai_explanation,
        )
    except Exception as e:
        print(f"PDF generation error: {e}")
        return f"PDF generation failed: {e}", 500

    # Safe filename banao
    safe_title = title.replace(" ", "_").replace("/", "-")
    pdf_filename = f"PlantDisease_Report_{safe_title}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=pdf_filename
    )
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
