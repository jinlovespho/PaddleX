import os 
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
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


# LOAD PADDLEOCR
# pipeline = create_pipeline(pipeline="OCR")
pipeline = create_pipeline(pipeline="./my_path/pho_ocr.yaml", device='gpu:3')
det_model_name = pipeline.config['SubModules']['TextDetection']['model_name']
rec_model_name = pipeline.config['SubModules']['TextRecognition']['model_name']


# set font - apt install fonts-nanum -y
font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", size=15)


# set data path
# data_path = '/media/dataset1/jinlovespho/ocr_plantynet/data'
root_path = '/media/dataset1/jinlovespho/ocr_plantynet/data/filtered'
imgs_path = f'{root_path}/images'
anns_path = f'{root_path}/anns'

imgs = sorted(os.listdir(imgs_path))
anns = sorted(os.listdir(anns_path))


SAVE_ROOT_PATH = './pho_vis'
save_gt_path = f'{SAVE_ROOT_PATH}/gt'
save_paddle_img_path = f'{SAVE_ROOT_PATH}/DET_{det_model_name}_REC_{rec_model_name}/pred_imgs'
save_paddle_ann_path = f'{SAVE_ROOT_PATH}/DET_{det_model_name}_REC_{rec_model_name}/pred_anns'


os.makedirs(save_gt_path, exist_ok=True)
os.makedirs(save_paddle_img_path, exist_ok=True)
os.makedirs(save_paddle_ann_path, exist_ok=True)


imgs=imgs[:10]
anns=anns[:10]


for img, ann in zip(imgs, anns):
    img_path = f'{imgs_path}/{img}'    
    ann_path = f'{anns_path}/{ann}'
    
    img_id = img_path.split('/')[-1].split('.')[0]
    
    
    GOOGLEOCR = True
    PADDLEOCR = True
    
    
    # GOOGLEOCR
    if GOOGLEOCR:
        
        gt_texts=[]
        gt_boxes=[]
        
        # Load original image to get dimensions
        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size

        # Create blank white canvas for visualization
        canvas_w = img_w
        canvas_h = img_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Load JSON annotation
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        if 'fullTextAnnotation' in ann['responses'][0].keys():
            pages = ann['responses'][0]['fullTextAnnotation']['pages'][0]
        blocks = pages['blocks']

        # Inside your drawing loop:
        for blk in blocks:
            for paragraph in blk['paragraphs']:
                for word in paragraph['words']:
                    text = ''.join([s['text'] for s in word['symbols']])
                    box_normalized = word['boundingBox']['normalizedVertices']
                    
                    # print(text)
                    gt_texts.append(text)
                    # breakpoint()

                    if any(('x' not in v or 'y' not in v) for v in box_normalized):
                        continue

                    box = [(int(v['x'] * img_w), int(v['y'] * img_h)) for v in box_normalized]

                    x0 = min(box[0][0], box[2][0])
                    y0 = min(box[0][1], box[2][1])
                    x1 = max(box[0][0], box[2][0])
                    y1 = max(box[0][1], box[2][1])
                    box_w = x1 - x0
                    box_h = y1 - y0

                    # Get font that fits inside the box
                    fitting_font = get_fitting_font(text, box_w, box_h)

                    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)
                    draw.text((x0 + 1, y0 + 1), text, font=fitting_font, fill=(0, 0, 0))
                    gt_boxes.append( ((x0,y0), (x1,y1)))

        # Combine original image and the canvas side-by-side
        combined = Image.new("RGB", (img_w * 2, img_h), (255, 255, 255))
        combined.paste(img_pil, (0, 0))
        combined.paste(canvas, (img_w, 0))

        # combined.save(f'./vis/googleocr/googleocr_{folder_id}_{img_id}.jpg')
        combined.save(f'{save_gt_path}/googleocr_{img_id}.jpg')
    
    # PADDLEOCR 
    if PADDLEOCR:
        
        output = pipeline.predict(
            input=img_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        
        for res in output:
            pred_texts = res['rec_texts']
            pred_boxes = [ (box.tolist()[0], box.tolist()[2]) for box in res['rec_polys']]

            # Save raw PaddleOCR output
            # res.save_to_img(save_path=f"{save_paddle_img_path}/paddleocr_{img_id}.jpg")
            res.save_to_json(save_path=f"{save_paddle_ann_path}/paddleocr_{img_id}.json")

            # Side-by-side visualization
            img_pil = Image.open(img_path)
            img_w, img_h = img_pil.size
            canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            for (p1, p2), text in zip(pred_boxes, pred_texts):
                x0, y0 = int(p1[0]), int(p1[1])
                x1, y1 = int(p2[0]), int(p2[1])
                box_w = x1 - x0
                box_h = y1 - y0

                fitting_font = get_fitting_font(text, box_w, box_h)
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)
                draw.text((x0 + 1, y0 + 1), text, font=fitting_font, fill=(0, 0, 0))

            # Combine side-by-side
            combined = Image.new("RGB", (img_w * 2, img_h), (255, 255, 255))
            combined.paste(img_pil, (0, 0))
            combined.paste(canvas, (img_w, 0))
            combined.save(f"{save_paddle_img_path}/paddleocr_{img_id}.jpg")
