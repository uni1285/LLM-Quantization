# L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models
Training and Evaluation codes for reproductions.


## Instructions
### Environment Settings
```bash
conda create -n l4q python=3.10
conda activate l4q
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt 
```

### Model Download and Data Preparation
```bash
# model from huggingface should be prepared to get a lit verision model
pip install -r requirements.txt
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
git clone https://huggingface.co/huggyllama/llama-7b

# huggingface model conversion to a lit model
python scripts/convert_hf_checkpoint.py --checkpoint_dir /path/to/hf_checkpoint  # outputs the converted checkpoint at checkpoints/dir/to/lit_checkpoint
python scripts/prepare_alpaca.py --destination_path data/llama --checkpoint_dir checkpoints/path/to/lit_checkpoint  # outputs the dataset at data/dir/to/alpaca-llama
```

### L4Q Fine-tuning
```bash
# LLaMA-1 example: the deafult parameters are set in a code
python finetune/llama.py --model_size 7b --w_bits 4 --learning_rate 5e-5 --max_iter 25600 --lsq_start 10 --out_dir out --data_dir data/alpaca_llama-1 --checkpoint_dir checkpoints/dir/to/lit_checkpoint
# LLaMA-2 example: the deafult parameters are set in a code
python finetune/lora.py --model_size 7b --w_bits 4 --learning_rate 1e-4 --max_iter 25600 --lsq_start 10 --out_dir out --data_dir data/alpaca_llama-2 --checkpoint_dir checkpoints/dir/to/lit_checkpoint
```

### Evaluation on lm_eval and MMLU
```bash
# LLaMA-1 example: the deafult parameters are set in a code.
python evaluate/llama.py --model_size 7b --w_bits 4 --lora_path out/l4q/llama-1/checkpoint_file.pth --checkpoint_dir checkpoints/dir/to/lit_checkpoint
# LLaMA-2 example: the deafult parameters are set in a code
python evaluate/lora.py --model_size 7b --w_bits 4 --lora_path out/l4q/llama-2/checkpoint_file.pth --checkpoint_dir checkpoints/dir/to/lit_checkpoint
```


## Acknowledgements
Our code is based on [lit-gpt](https://github.com/Lightning-AI/litgpt), [alpaca-dataset](https://github.com/gururise/AlpacaDataCleaned).
