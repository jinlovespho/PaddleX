python main.py -c paddlex/configs/modules/text_recognition/korean_PP-OCRv3_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/media/dataset1/jinlovespho/ocr_plantynet/data/rec_training_data \
    -o Global.output=./pho_train_result/pho_rec/train_gpu0123_ep10_bs256_lr1e-3 \
    -o Global.device=gpu:0,1,2,3