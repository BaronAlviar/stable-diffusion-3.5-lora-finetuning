# Modular Stable Diffusion 3 LoRA Fine-Tuning

This project provides a modular and configurable framework for fine-tuning the [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) or [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) model using Low-Rank Adaptation (LoRA). The code is structured to be clean, readable, and easily extensible for custom experiments.

## Features

- **Modular Structure**: Code is separated into logical components for better maintainability.
- **Configuration-Driven**: All hyperparameters and paths are managed via a single `configs/training_config.yaml` file.
- **Accelerate Integration**: Leverages Hugging Face's `accelerate` for seamless multi-GPU and mixed-precision training.
- **Memory Efficient**: Supports 4-bit quantization via `bitsandbytes` and gradient checkpointing.
- **Rectified Flow (RF) Loss**: Implements a training objective based on Rectified Flow principles.
- **Easy Inference**: Includes a dedicated `inference.py` script to generate images from your fine-tuned LoRA adapters.

## Project Structure

```
lora_fine_tuning_project/
├── configs/
│   └── training_config.yaml      # All training parameters
├── src/
│   ├── config_loader.py          # Loads the YAML config
│   ├── data_processing.py        # Dataset class and metadata loading
│   ├── model_setup.py            # Handles model and LoRA setup
│   ├── trainer.py                # The main training loop class
│   └── utils.py                  # Helper functions
├── train.py                        # Main script to start training
├── inference.py                    # Script to generate images with a trained LoRA
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```


## Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd lora_fine_tuning_project
```

### 2. Install Dependencies
It's recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Configure Accelerator (Crucial Step)

The training script relies on Hugging Face `accelerate`. To ensure it uses the correct settings (like mixed precision), you need to configure it once per machine.

Run the following command in your terminal:
```bash
accelerate config
```
This will launch an interactive setup wizard. For most single-GPU setups (like Google Colab or a standard desktop), you can follow the answers below.

<details>
<summary><strong>Click to see the `accelerate config` example dialogue</strong></summary>

```text
In which compute environment are you running?
[0] This machine
[1] AWS (Amazon SageMaker)
-> 0 (This machine)

Which type of machine are you using?
[0] No distributed training
[1] multi-CPU
[2] multi-GPU
...
-> 0 (No distributed training, if you are on a single GPU)

Do you want to run your training on CPU only? [yes/NO]:
-> NO

Do you wish to optimize your script with torch dynamo? [yes/NO]:
-> NO

Do you want to use DeepSpeed? [yes/NO]:
-> NO

What GPU(s) (by id) should be used for training on this machine? (e.g. '0,1,2,3', 'all') [all]:
-> all (or 0)

Would you like to enable numa efficiency? [yes/NO]:
-> NO (This is safe for most systems)

Do you wish to use mixed precision?
[0] no
[1] fp16
[2] bf16
...
-> 2 (bf16, to match our project's config)

```
After answering, `accelerate` will save a configuration file, and you won't see warnings when you run `accelerate launch`.
</details>

### 4. Log in to Hugging Face (Crucial Step)

**⚠️ IMPORTANT:** The Stable Diffusion 3 models are gated. To download them, you must first agree to the license terms on the model's Hugging Face page and then log in.

1.  Go to the [Stable Diffusion 3.5 Medium model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) or [Stable Diffusion 3.5 Large model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) and accept the terms.
2.  Get an access token from your Hugging Face account settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3.  Run the following command in your terminal and paste your token when prompted:
    ```bash
    huggingface-cli login
    ```

### 5. Prepare Your Dataset

The training script requires a specific dataset format. For a detailed guide on creating a high-quality dataset from scratch, please refer to the **[Text-to-Image Dataset Preparation Toolkit](https://github.com/gokhaneraslan/text_to_image_dataset_toolkit)**.

For a quick test, ensure your dataset directory contains an `images/` subdirectory and a `metadata.jsonl` file.

## Usage

### Training

To start the fine-tuning process, run:
```bash
accelerate launch train.py --config_path configs/training_config.yaml
```

### Inference

Once training is complete, use `inference.py` to generate images:
```bash
python inference.py \
    --lora_checkpoint_dir "path/to/your/checkpoint-final" \
    --prompt "A ghibli-style painting of a robot in a field of flowers" \
    --output_path "outputs/images/my_first_lora_image.png"
```
Run `python inference.py --help` for all available options.
