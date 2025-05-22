python main.py -c paddlex/configs/modules/text_recognition/korean_PP-OCRv3_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/media/dataset1/jinlovespho/ocr_plantynet/data/pho_vis_rec \
    -o Global.output=./output/pho_rec/re_train_gpu01_ep8_bs256_lr1e-4 \
    -o Global.device=gpu:0,1