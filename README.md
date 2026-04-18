# RAG
Using Rag to extract context based on query from a pdf


1- Create a hugging face api key. Otherwise, it won't be able to download the trained llm. Look for RAGPipeLine.py class. Replase HF_KEY = "YOUR_TOKEN" with your token.


2- load_pdf(self, pdf_path) function is where I have parsed the Quran the way I need it. You gotta modify according to your need. 


3- In the main.py, I loaded the pdf directly from the root directory of the project. document = "quran-english.pdf". chande your pdf or write your own method to connect with jenkins to retrieve the data.


4- Since my target was to get a precise answer, my new_max_token = 2048. It takes 3-5 minutes to answer your one question. But if you want small answer, then just change the max_new_token to 128 or 256 or something like that


5- Since my GPU does not support that large data, I have only did it with CPU. That's one of the reason it takes a while to generate the answers.



**Environment:**

I have used Ubuntu 24.04 with virtual env since, running BitAndByteConfig for quantization only supports python 3.11 or lower. 3.12 won't work. In order to install python 3.11 virtual environment without messing up the existing setup, you need to follow below instructions:


This installs Python 3.11 alongside your existing version:

sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.11 python3.11-venv python3.11-dev -y

python3.11 --version


Create a directory where you want the install the virtual env:

mkdir -p ~/projects/my_py311_project
cd ~/projects/my_py311_project


Create a virtual Invironment:

python3.11 -m venv venv

Actiavte is:

source venv/bin/activate

Upgrade pip:

pip install --upgrade pip



If you wanna use cuda or gpu, install all necessary packages to run this code. I have used cuda 11.8 which works best with python 3.11:

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

pip install \
  transformers accelerate datasets \
  sentencepiece safetensors \
  numpy pandas matplotlib scikit-learn \
  jupyterlab ipykernel tqdm

Test if you have the access to cuda. Make sure your venv is active and your interpreter and configuration is pointing to the right venv that you just created. This can be managed in python settings:


import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.rand(3000, 3000, device="cuda")
    y = torch.matmul(x, x)
    print("GPU computation successful:", y.shape)
else:
    print("Running on CPU")

After you are done, deactivate the venv that you have activated to run this project so that it can start using the system python for the rest of the project in your machine.

deactivate


**One more important thing:**

faiss-gpu will not install in this setup since I am not using conda. Therefore you gotta install faiss-gpu. This would cost 3-5 mins to create the vector db at the first place. But once it's done, there is no more waiting. So use this command to install faiss-cpu:

pip install faiss-cpu
