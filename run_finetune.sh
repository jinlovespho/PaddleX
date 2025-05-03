CUDA_VISIBLE_DEVICES=3 python main.py       -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
                                            -o Global.mode=train \
                                            -o Global.dataset_dir=your/dataset_dir