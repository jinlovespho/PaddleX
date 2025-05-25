      
## Cooperation with PlantyNet (PaddleX fine-tuned for korean)


### Updates
- **2025.05.25** Training recognition model

### 📌 TODO
- [ ] Train the recognizer
- [ ] Train the detector

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
#### 1. Test Dataset Structure
```
data/
    └── 
        └── 0000001.png # GT images, (512, 512, 3)
        └── ...
    └── lr
        └── 0000001.png # LR images, (512, 512, 3)
        └── ...
    └── tag
        └── 0000001.txt # tag prompts
        └── ...
```


#### 2. Test eval script 

```
bash run_pho_test.sh
```

## Training

#### Step1: Download the pretrained models
Download the pretrained [SD-2-base models](https://huggingface.co/stabilityai/stable-diffusion-2-base) and [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth). You can put them into `preset/models`.

#### Step2: Prepare training data
We pre-prepare training data pairs for the training process, which would take up some memory space but save training time. We train the DAPE with [COCO](https://cocodataset.org/#home) and train the SeeSR with LSDIR+FFHQ10k.

For making paired data when training DAPE, you can run:

```
python utils_data/make_paired_data_DAPE.py \
--gt_path PATH_1 PATH_2 ... \
--save_dir preset/datasets/train_datasets/training_for_dape \
--epoch 1
```
For making paired data when training SeeSR, you can run:
```
python utils_data/make_paired_data.py \
--gt_path PATH_1 PATH_2 ... \
--save_dir preset/datasets/train_datasets/training_for_dape \
--epoch 1
```

- `--gt_path` the path of gt images. If you have multi gt dirs, you can set it as `PATH1 PATH2 PATH3 ...`
- `--save_dir` the path of paired images 
- `--epoch` the number of epoch you want to make

The difference between `make_paired_data_DAPE.py` and `make_paired_data.py` lies in that `make_paired_data_DAPE.py` resizes the entire image to a resolution of 512, while `make_paired_data.py` randomly crops a sub-image with a resolution of 512.


Once the degraded data pairs are created, you can base them to generate tag data by running `utils_data/make_tags.py`.

The data folder should be like this:
```
your_training_datasets/
    └── gt
        └── 0000001.png # GT images, (512, 512, 3)
        └── ...
    └── lr
        └── 0000001.png # LR images, (512, 512, 3)
        └── ...
    └── tag
        └── 0000001.txt # tag prompts
        └── ...
```

#### Step3: Training for DAPE
Please specify the DAPE training data path at `line 13` of `basicsr/options/dape.yaml`, then run the training command:
```
python basicsr/train.py -opt basicsr/options/dape.yaml
```
You can modify the parameters in `dape.yaml` to adapt to your specific situation, such as the number of GPUs, batch size, optimizer selection, etc. For more details, please refer to the settings in Basicsr. 
#### Step4: Training for SeeSR
```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/seesr" \
--root_folders 'preset/datasets/training_datasets' \
--ram_ft_path 'preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 
--dataloader_num_workers=0 \
--checkpointing_steps=10000 
```
- `--pretrained_model_name_or_path` the path of pretrained SD model from Step 1
- `--root_folders` the path of your training datasets from Step 2
- `--ram_ft_path` the path of your DAPE model from Step 3


The overall batch size is determined by num of `CUDA_VISIBLE_DEVICES`, `--train_batch_size`, and `--gradient_accumulation_steps` collectively. If your GPU memory is limited, you can consider reducing `--train_batch_size` while increasing `--gradient_accumulation_steps`.


## Acknowledgments
This project is based on [PaddleX](https://github.com/PaddlePaddle/PaddleX). Thanks for their awesome works 

## Contact
If you have any questions, please feel free to contact: `msjchr@korea.ac.kr`



# PaddleX - plantynet


# Preparation
```
pip install konlpy 
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
pip install numpy==1.26.4 

```
