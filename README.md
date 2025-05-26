      
## Cooperation with PlantyNet (PaddleX fine-tuned for korean)


## ⚙️ Dependencies and Installation
```
# clone repo 
git clone https://github.com/jinlovespho/PaddleX.git -b pho
cd PaddleX

# create conda env
conda create -n pho_paddlex python=3.9 -y 
conda activate pho_paddlex 

# install paddle libraries
pip install paddlepaddle 
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install -r requirements.txt
pip install -e .
paddlex --install PaddleOCR

# install additional libraries
pip install albucore==0.0.16
pip install konlpy 
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
pip install numpy==1.26.4 
```

## Evaluation
#### Run evaluation script
```
bash run_pho_test.sh
```

#### Evaluation script arguments
```
python pho_test.py \
    --test_imgs_path /path/to/test/imgs \
    --test_anns_path /path/to/test/anns \
    --save_root_path /path/to/save/results \
    --gpu 0 


# example usage 
python pho_test.py \
    --test_imgs_path ocr_dataset/test_images \
    --test_anns_path ocr_dataset/test_anns \
    --save_root_path ./ocr_results \
    --gpu 3


# example test dataset structure 
ocr_dataset/
    └── test_images/
        └── 000000030633-01_007.png 
        └── ...
    └── test_anns/
        └── 000000030633-01_007.json
        └── ...
```


## Acknowledgments
This project is based on [PaddleX](https://github.com/PaddlePaddle/PaddleX). Thanks for their awesome works 

## Contact
If you have any questions, please feel free to contact: `msjchr@korea.ac.kr`
