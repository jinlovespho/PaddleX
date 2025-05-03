import os 
from paddlex import create_pipeline


# pipeline = create_pipeline(pipeline="OCR")
pipeline = create_pipeline(pipeline="./my_path/OCR.yaml", device='gpu')

det_model_name = pipeline.config['SubModules']['TextDetection']['model_name']
rec_model_name = pipeline.config['SubModules']['TextRecognition']['model_name']

img_paths=sorted([f'/media/dataset1/jinlovespho/ocr_plantynet/google_ocr_bbox/000000030633-01/{img_name}' for img_name in os.listdir('/media/dataset1/jinlovespho/ocr_plantynet/google_ocr_bbox/000000030633-01') if img_name.endswith('png')])

output = pipeline.predict(
    input=img_paths,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

for res in output:
    # breakpoint()
    res.print()
    res.save_to_img(save_path=f"./vis/{det_model_name}_{rec_model_name}")
    res.save_to_json(save_path=f"./vis/{det_model_name}_{rec_model_name}")