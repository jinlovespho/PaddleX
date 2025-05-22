python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples \
    -o Global.output=./output/demo_det_val \
    -o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams