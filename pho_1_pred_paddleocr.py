import os 
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
DATA_PATH = '/media/dataset1/jinlovespho/ocr_plantynet/data/filtered_tmp'

imgs = sorted(os.listdir(f'{DATA_PATH}/images'))
anns = sorted(os.listdir(f'{DATA_PATH}/anns'))
json_anns = sorted(os.listdir(f'{DATA_PATH}/json_anns'))


img_paths = sorted([f'{DATA_PATH}/images/{img}' for img in imgs])
ann_paths = sorted([f'{DATA_PATH}/anns/{ann}' for ann in anns])
json_ann_paths = sorted([f'{DATA_PATH}/json_anns/{json_ann}' for json_ann in json_anns])


print('imgs:', len(imgs))
print('anns:', len(anns))
print('json_anns:', len(json_anns))


output = pipeline.predict(
    input=img_paths,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


for res in output:
    img_info = res['input_path'].split('/')[-1].split('.')[0]
    # breakpoint()
    res.print()
    # res.save_to_img(save_path=f"./vis/DET_{det_model_name}_REC_{rec_model_name}/paddleocr_{folder_id}_{img_id}.jpg",)
    # res.save_to_json(save_path=f"./vis/DET_{det_model_name}_REC_{rec_model_name}/paddleocr_{folder_id}_{img_id}.json",)
    res.save_to_img(save_path=f"./vis/paddleocr/DET_{det_model_name}_REC_{rec_model_name}/pred_img/{img_info}.jpg",)
    res.save_to_json(save_path=f"./vis/paddleocr/DET_{det_model_name}_REC_{rec_model_name}/pred_ann/{img_info}.json",)

