import os
import json
from shapely.geometry import Polygon
from shapely.errors import TopologicalError

IOU_THRESHOLD = 0.5

def polygon_from_points(points):
    try:
        return Polygon(points)
    except TopologicalError:
        return Polygon()

def compute_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def load_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    return [(polygon_from_points(entry['points']), entry['transcription']) for entry in gt]

def load_pred(pred_path):
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    return [(polygon_from_points(poly), text) for poly, text in zip(pred['rec_polys'], pred['rec_texts'])]

def evaluate(gt_list, pred_list):
    matched_gt = set()
    matched_pred = set()
    correct_texts = 0
    incorrect_texts = 0

    correct_samples = []
    incorrect_samples = []

    gt_polys = [p[0] for p in gt_list]
    gt_texts = [p[1] for p in gt_list]

    pred_polys = [p[0] for p in pred_list]
    pred_texts = [p[1] for p in pred_list]

    for pi, (ppoly, ptext) in enumerate(zip(pred_polys, pred_texts)):
        best_iou = 0
        best_gi = -1
        for gi, (gpoly, gtext) in enumerate(zip(gt_polys, gt_texts)):
            if gi in matched_gt:
                continue
            iou = compute_iou(ppoly, gpoly)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= IOU_THRESHOLD:
            matched_pred.add(pi)
            matched_gt.add(best_gi)
            gt_text = gt_texts[best_gi].strip().lower()
            pred_text = pred_texts[pi].strip().lower()
            if pred_text == gt_text:
                correct_texts += 1
                correct_samples.append((gt_texts[best_gi], pred_texts[pi]))
            else:
                incorrect_texts += 1
                incorrect_samples.append((gt_texts[best_gi], pred_texts[pi]))

    tp = len(matched_pred)
    fp = len(pred_polys) - tp
    fn = len(gt_polys) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    rec_acc = correct_texts / tp if tp > 0 else 0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'recognition_accuracy': rec_acc,
        'correct_samples': correct_samples,
        'incorrect_samples': incorrect_samples
    }

def evaluate_folders(gt_folder, pred_folder):
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".json")])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".json")])
    common_files = set(gt_files).intersection(pred_files)

    if not common_files:
        print("No matching JSON files found.")
        return

    metrics_sum = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'recognition_accuracy': 0
    }
    num_files = 0

    for f_idx, fname in enumerate(sorted(common_files)):
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        gt = load_gt(gt_path)
        pred = load_pred(pred_path)
        metrics = evaluate(gt, pred)
        
        print(f"\n{f_idx+1}. File: {fname}")
        for k, v in metrics.items():
            if k not in ['correct_samples', 'incorrect_samples']:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("  ✓ Correct Predictions (up to 5):")
        for gt_text, pred_text in metrics['correct_samples'][:5]:
            print(f"    GT: '{gt_text}' | Pred: '{pred_text}'")

        print("  ✗ Incorrect Predictions (up to 5):")
        for gt_text, pred_text in metrics['incorrect_samples'][:5]:
            print(f"    GT: '{gt_text}' | Pred: '{pred_text}'")

        for k in metrics_sum:
            metrics_sum[k] += metrics[k]
        num_files += 1

    print("\n===== FINAL AVERAGED METRICS =====")
    print("\n--- Detection Metrics ---")
    print(f"Total files evaluated: {num_files}")
    print(f"Total True Positives: {metrics_sum['true_positives']}")
    print(f"Total False Positives: {metrics_sum['false_positives']}")
    print(f"Total False Negatives: {metrics_sum['false_negatives']}")
    avg_precision = metrics_sum['precision'] / num_files if num_files > 0 else 0
    avg_recall = metrics_sum['recall'] / num_files if num_files > 0 else 0
    avg_f1 = metrics_sum['f1_score'] / num_files if num_files > 0 else 0
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1 Score:  {avg_f1:.4f}")

    print("\n--- Recognition Metric ---")
    avg_recognition_accuracy = metrics_sum['recognition_accuracy'] / num_files if num_files > 0 else 0
    print(f"Average Recognition Accuracy: {avg_recognition_accuracy:.4f}")

if __name__ == "__main__":
    
    paddleocr_model_dir = 'pho_vis/DET_PP-OCRv4_server_det_REC_korean_PP-OCRv3_mobile_rec'

    
    SAVE_RESULT_PATH = f"{paddleocr_model_dir}/eval_metric_result.txt"
    PRED_JSON_PATH = f"pho_vis/DET_PP-OCRv4_server_det_REC_korean_PP-OCRv3_mobile_rec/pred_anns"
    GT_JSON_PATH = "/media/dataset1/jinlovespho/ocr_plantynet/data/filtered_tmp/json_anns"
    
    
    with open(SAVE_RESULT_PATH, "w", encoding="utf-8") as f:
        # redirect prints to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            evaluate_folders(GT_JSON_PATH, PRED_JSON_PATH)
        finally:
            sys.stdout = original_stdout
    print(f"Results saved to {SAVE_RESULT_PATH}")
