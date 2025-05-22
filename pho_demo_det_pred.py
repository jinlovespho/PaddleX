from paddlex import create_model

# model = create_model(model_name="PP-OCRv5_mobile_det")

model = create_model(model_name='PP-OCRv4_mobile_det',
                     model_dir='output/demo_det_train/best_accuracy/inference')


output = model.predict("tmp1.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")