conda create -n unlearn python=3.11 -y && conda activate unlearn

pip install -r requirements.txt
pip install -e . --no-deps
pip install lm_eval==0.4.8
pip install tiktoken

# git init  # needed for path detection