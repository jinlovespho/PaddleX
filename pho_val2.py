import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddlex import create_pipeline


def get_fitting_font(text, box_width, box_height, font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    max_font_size = min(int(box_height), 20)
    for size in range(max_font_size, 4, -1):
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= box_width and text_h <= box_height:
            return font
    return ImageFont.truetype(font_path, 5)

def merge_words_by_symbol_break(words, img_w, img_h):
    merged_texts = []
    merged_boxes = []

    current_text = ""
    current_vertices = []

    def norm2abs(vertex):
        return int(vertex['x'] * img_w), int(vertex['y'] * img_h)

    def merge_bbox(vertices_list):
        xs = [v['x'] for box in vertices_list for v in box if 'x' in v]
        ys = [v['y'] for box in vertices_list for v in box if 'y' in v]
        if not xs or not ys:
            return 0, 0, 0, 0  # fallback safe box if no valid coords
        return (
            int(min(xs) * img_w), int(min(ys) * img_h),
            int(max(xs) * img_w), int(max(ys) * img_h)
        )

    for word in words:
        symbols = word.get("symbols", [])
        bbox = word.get("boundingBox", {}).get("normalizedVertices", [])
        if len(bbox) < 4 or any(('x' not in v or 'y' not in v) for v in bbox):
            continue

        for i, symbol in enumerate(symbols):
            current_text += symbol['text']
        current_vertices.append(bbox)

        # Check if current symbol has a break
        last_symbol = symbols[-1]
        break_type = last_symbol.get("property", {}).get("detectedBreak", {}).get("type", None)
        if break_type in ("SPACE", "LINE_BREAK", "EOL_SURE_SPACE") or word == words[-1]:
            # Finalize current merged word
            x0, y0, x1, y1 = merge_bbox(current_vertices)
            merged_texts.append(current_text)
            merged_boxes.append(((x0, y0), (x1, y1)))
            current_text = ""
            current_vertices = []

    return merged_texts, merged_boxes

def iou(boxA, boxB):
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
    boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def evaluate(gt_boxes, gt_texts, pred_boxes, pred_texts, iou_thresh=0.5):
    matched_gt = set()
    matched_pred = set()
    correct_text_count = 0

    for i, (p_box, p_text) in enumerate(zip(pred_boxes, pred_texts)):
        best_iou = 0
        best_j = -1
        for j, g_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou_val = iou(p_box, g_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            matched_pred.add(i)
            if p_text.strip() == gt_texts[best_j].strip():
                correct_text_count += 1

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Correct Text Matches': correct_text_count,
        'Detection Precision': tp / (tp + fp + 1e-6),
        'Detection Recall': tp / (tp + fn + 1e-6),
        'Recognition Accuracy': correct_text_count / (tp + 1e-6)
    }


# --- MAIN PIPELINE ---

pipeline = create_pipeline(pipeline="./my_path/pho_ocr.yaml", device='gpu:3')
det_model_name = pipeline.config['SubModules']['TextDetection']['model_name']
rec_model_name = pipeline.config['SubModules']['TextRecognition']['model_name']

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = ImageFont.truetype(font_path, size=15)

root_path = '/media/dataset1/jinlovespho/ocr_plantynet/data/filtered'
imgs_path = f'{root_path}/images'
anns_path = f'{root_path}/anns'

imgs = sorted(os.listdir(imgs_path))[:10]
anns = sorted(os.listdir(anns_path))[:10]

SAVE_ROOT_PATH = './pho_vis2'
save_gt_path = f'{SAVE_ROOT_PATH}/gt'
save_paddle_img_path = f'{SAVE_ROOT_PATH}/DET_{det_model_name}_REC_{rec_model_name}/pred_imgs'
save_paddle_ann_path = f'{SAVE_ROOT_PATH}/DET_{det_model_name}_REC_{rec_model_name}/pred_anns'

os.makedirs(save_gt_path, exist_ok=True)
os.makedirs(save_paddle_img_path, exist_ok=True)
os.makedirs(save_paddle_ann_path, exist_ok=True)

all_tp, all_fp, all_fn, all_correct = 0, 0, 0, 0

for img, ann in zip(imgs, anns):
    img_path = os.path.join(imgs_path, img)
    ann_path = os.path.join(anns_path, ann)
    img_id = os.path.splitext(img)[0]

    gt_texts, gt_boxes = [], []

    # --- Google OCR Ground Truth ---
    with open(ann_path, 'r') as f:
        ann_data = json.load(f)
    if 'fullTextAnnotation' in ann_data['responses'][0]:
        blocks = ann_data['responses'][0]['fullTextAnnotation']['pages'][0]['blocks']
        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size
        canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        gt_texts, gt_boxes = [], []

        for blk in blocks:
            for para in blk.get("paragraphs", []):
                merged_texts, merged_boxes = merge_words_by_symbol_break(para.get("words", []), img_w, img_h)
                gt_texts.extend(merged_texts)
                gt_boxes.extend(merged_boxes)

        for text, ((x0, y0), (x1, y1)) in zip(gt_texts, gt_boxes):
            draw.rectangle([x0, y0, x1, y1], outline='black', width=2)
            draw.text((x0, y0), text, fill='black', font=get_fitting_font(text, x1 - x0, y1 - y0))

        # Combine original image and the canvas side-by-side
        combined = Image.new("RGB", (img_w * 2, img_h), (255, 255, 255))
        combined.paste(img_pil, (0, 0))
        combined.paste(canvas, (img_w, 0))
        combined.save(f'{save_gt_path}/googleocr_{img_id}.jpg')

    # --- PaddleOCR Predictions ---
    output = pipeline.predict(
        input=img_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    pred_texts, pred_boxes = [], []
    # Side-by-side visualization
    img_pil = Image.open(img_path)
    img_w, img_h = img_pil.size
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for res in output:
        pred_texts.extend(res['rec_texts'])
        pred_boxes.extend([(box.tolist()[0], box.tolist()[2]) for box in res['rec_polys']])
        res.save_to_json(save_path=f"{save_paddle_ann_path}/paddleocr_{img_id}.json")

    # --- Evaluate ---
    result = evaluate(gt_boxes, gt_texts, pred_boxes, pred_texts)
    all_tp += result['TP']
    all_fp += result['FP']
    all_fn += result['FN']
    all_correct += result['Correct Text Matches']
    
    matched_gt = set()
    color_map = []

    for i, (p_box, p_text) in enumerate(zip(pred_boxes, pred_texts)):
        best_iou = 0
        best_j = -1
        for j, g_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou_val = iou(p_box, g_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_iou >= 0.3:
            matched_gt.add(best_j)
            if p_text.strip() == gt_texts[best_j].strip():
                color = 'green'  # TP + correct recognition
            else:
                color = 'orange'  # TP + wrong recognition
        else:
            color = 'red'  # FP

        color_map.append(color)

    # Draw each box with its assigned color
    for (p0, p1), text, color in zip(pred_boxes, pred_texts, color_map):
        draw.rectangle([p0, p1], outline=color, width=2)
        font = get_fitting_font(text, p1[0] - p0[0], p1[1] - p0[1])
        draw.text((p0[0], p0[1]), text, fill=color, font=font)

    # Combine side-by-side
    combined = Image.new("RGB", (img_w * 2, img_h), (255, 255, 255))
    combined.paste(img_pil, (0, 0))
    combined.paste(canvas, (img_w, 0))
    combined.save(f"{save_paddle_img_path}/paddleocr_{img_id}.jpg")
    
    # --- Detailed Logging ---
    log_path = f"{SAVE_ROOT_PATH}/DET_{det_model_name}_REC_{rec_model_name}/logs"
    os.makedirs(log_path, exist_ok=True)
    with open(f"{log_path}/log_{img_id}.txt", "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Detailed Recognition Results for {img_id} ===\n\n")
        log_file.write(f"--- Predictions ---\n")
        for i, (p_box, p_text) in enumerate(zip(pred_boxes, pred_texts)):
            match_status = color_map[i]
            if match_status == 'green':
                note = "⭕️"
            elif match_status == 'orange':
                note = "❌"
            else:
                note = "❓"
            log_file.write(f"[{i:02d}] | {note} Text: '{p_text}' | Box: {p_box}\n")

        log_file.write(f"\n--- Ground Truths ---\n")
        for j, (gt_text, gt_box) in enumerate(zip(gt_texts, gt_boxes)):
            log_file.write(f"[{j:02d}] GT Text: '{gt_text}' | Box: {gt_box}\n")

    print(f"[{img_id}] Precision: {result['Detection Precision']:.3f}, "
          f"Recall: {result['Detection Recall']:.3f}, "
          f"RecAcc: {result['Recognition Accuracy']:.3f}")
    print(f"→ Detailed log saved to {log_path}/log_{img_id}.txt")

    # print(f"[{img_id}] Precision: {result['Detection Precision']:.3f}, "
    #       f"Recall: {result['Detection Recall']:.3f}, "
    #       f"RecAcc: {result['Recognition Accuracy']:.3f}")

# --- Final Summary ---
precision = all_tp / (all_tp + all_fp + 1e-6)
recall = all_tp / (all_tp + all_fn + 1e-6)
rec_acc = all_correct / (all_tp + 1e-6)

print("\n=== Overall Evaluation ===")
print(f"Total True Positives: {all_tp}")
print(f"Total False Positives: {all_fp}")
print(f"Total False Negatives: {all_fn}")
print(f"Total Correct Text Matches: {all_correct}")
print(f"Detection Precision: {precision:.3f}")
print(f"Detection Recall: {recall:.3f}")
print(f"Recognition Accuracy: {rec_acc:.3f}")
