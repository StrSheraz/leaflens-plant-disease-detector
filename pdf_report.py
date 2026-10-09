import io, base64
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage

DARK_GREEN  = colors.HexColor("#042908")
LIGHT_GREEN = colors.HexColor("#4ECC5A")
MID_GREEN   = colors.HexColor("#063d0c")
ORANGE      = colors.HexColor("#E07B00")
LIGHT_GRAY  = colors.HexColor("#f8f9fa")
TEXT_DARK   = colors.HexColor("#1a1a1a")
TEXT_GRAY   = colors.HexColor("#555555")
WHITE       = colors.white

def generate_pdf_report(title, description, prevent, original_image_path, gradcam_b64, confidence=None, xai_explanation=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    style_h1 = ParagraphStyle("H1", parent=styles["Normal"], fontSize=20, leading=26, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
    style_h2 = ParagraphStyle("H2", parent=styles["Normal"], fontSize=13, leading=18, textColor=DARK_GREEN, fontName="Helvetica-Bold", spaceAfter=4)
    style_label = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9, leading=12, textColor=TEXT_GRAY, fontName="Helvetica-Bold", spaceAfter=2)
    style_body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=15, textColor=TEXT_DARK, fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=6)
    style_small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=11, textColor=TEXT_GRAY, fontName="Helvetica", alignment=TA_CENTER)

    story = []

    header_data = [[Paragraph("Plant Disease Detection Report", style_h1)]]
    header_table = Table(header_data, colWidths=[doc.width])
    header_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),DARK_GREEN),("TOPPADDING",(0,0),(-1,-1),18),("BOTTOMPADDING",(0,0),(-1,-1),18)]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    now = datetime.now().strftime("%B %d, %Y  |  %I:%M %p")
    meta_data = [[Paragraph(f"Generated: {now}", style_small), Paragraph("LeafHealth Detector - AI Powered Analysis", style_small)]]
    meta_table = Table(meta_data, colWidths=[doc.width/2, doc.width/2])
    meta_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT_GRAY),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    disease_style = ParagraphStyle("Disease", parent=styles["Normal"], fontSize=15, leading=20, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
    conf_text = f"  |  Confidence: {confidence}%" if confidence else ""
    disease_data = [[Paragraph(f"Detected: {title}{conf_text}", disease_style)]]
    disease_table = Table(disease_data, colWidths=[doc.width])
    disease_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),MID_GREEN),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10)]))
    story.append(disease_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Visual Analysis", style_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=LIGHT_GREEN, spaceAfter=8))

    img_w = (doc.width / 2) - 0.5*cm
    img_h = 6*cm

    orig_img_obj = None
    try:
        orig_img_obj = RLImage(str(original_image_path), width=img_w, height=img_h)
    except:
        pass

    gcam_img_obj = None
    try:
        gcam_bytes = base64.b64decode(gradcam_b64)
        gcam_buf = io.BytesIO(gcam_bytes)
        gcam_img_obj = RLImage(gcam_buf, width=img_w, height=img_h)
    except:
        pass

    orig_label_style = ParagraphStyle("OL", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)
    label_table = Table([[Paragraph("Original Image", orig_label_style), Paragraph("Grad-CAM Heatmap", orig_label_style)]], colWidths=[img_w+0.5*cm, img_w+0.5*cm])
    label_table.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),DARK_GREEN),("BACKGROUND",(1,0),(1,0),colors.HexColor("#8B4513")),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.append(label_table)

    img_table = Table([[orig_img_obj or Paragraph("N/A", style_small), gcam_img_obj or Paragraph("N/A", style_small)]], colWidths=[img_w+0.5*cm, img_w+0.5*cm])
    img_table.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("BACKGROUND",(0,0),(-1,-1),LIGHT_GRAY),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),("LINEBELOW",(0,0),(-1,-1),2,LIGHT_GREEN)]))
    story.append(img_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Disease Description", style_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=LIGHT_GREEN, spaceAfter=8))
    story.append(Paragraph(description, style_body))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Prevention & Treatment Steps", style_h2))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=8))
    steps = [s.strip() for s in prevent.replace("\r","").split("\n") if s.strip()]
    if len(steps) <= 1:
        steps = [s.strip()+"." for s in prevent.split(".") if s.strip()]
    for i, step in enumerate(steps[:10], 1):
        story.append(Paragraph(f"<b>{i}.</b>  {step}", style_body))
    story.append(Spacer(1, 0.4*cm))

    if xai_explanation:
        story.append(Paragraph("AI Explainability (XAI) Analysis", style_h2))
        story.append(HRFlowable(width="100%", thickness=2, color=LIGHT_GREEN, spaceAfter=8))
        for label, content in [("Disease Overview", xai_explanation.get("overview","")),("What Heatmap Shows", xai_explanation.get("heatmap_focus","")),("Why Model Focused There", xai_explanation.get("model_reasoning","")),("Recommended Action", xai_explanation.get("recommended_action",""))]:
            if content:
                story.append(Paragraph(label, style_label))
                story.append(Paragraph(content, style_body))
                story.append(Spacer(1, 0.2*cm))

    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREEN, spaceBefore=8, spaceAfter=6))
    story.append(Paragraph("This report was generated automatically by the LeafHealth AI Detection System. Results should be verified by an agricultural expert.", style_small))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
