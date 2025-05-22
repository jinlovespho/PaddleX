conda create -n pho_paddlex2 python=3.9 -y 
conda activate pho_paddlex2 
pip install paddlepaddle 
# https://www.paddlepaddle.org.cn/en
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
cd PaddleX
pip install -r requirements.txt
pip install -e .
paddlex --install PaddleOCR
pip install albucore==0.0.16