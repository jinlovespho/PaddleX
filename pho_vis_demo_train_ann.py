import os 
import cv2 
import json 
import numpy as np 

img_path = './dataset/ocr_det_dataset_examples/images'
imgs = os.listdir(img_path)


train_imgs = sorted([train_img for train_img in imgs if train_img.startswith('train')])
train_ann_path ='./dataset/ocr_det_dataset_examples/train.txt'
with open(train_ann_path, 'r') as f:
    train_anns = f.readlines()
    train_anns = sorted(train_anns)
    
    
for train_img, train_ann in zip(train_imgs, train_anns):
    train_img_path = os.path.join(img_path, train_img)
    img = cv2.imread(train_img_path)
    h, w, c = img.shape
    print(h, w, c)
    
    # get the ann
    train_img_info = train_ann.split('\t')[0]
    img_name = train_img_info.split('/')[-1]
    train_ann = json.loads(train_ann.split('\t')[1])
    
    for ann in train_ann:
        # get the ann
        text = ann['transcription']
        poly_four = ann['points']
        
        cv2.polylines(img, [np.array(poly_four)], True, (0, 255, 0), 2)
    
    cv2.imwrite(f'./vis/demo_det_train_data/{img_name}.jpg', img)

breakpoint()