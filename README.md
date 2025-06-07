# 🎨 Stable Diffusion 3 LoRA Fine-Tuning Framework

A comprehensive, modular framework for fine-tuning Stable Diffusion 3.5 models using Low-Rank Adaptation (LoRA). This repository enables you to create custom AI image generators tailored to your specific artistic style, objects, or concepts.

## 🤔 What is Text-to-Image Generation?

Text-to-image generation is a revolutionary AI technology that creates images from written descriptions. Instead of drawing or photographing, you simply describe what you want to see, and the AI generates a corresponding image.

**Examples:**
- Input: "A serene mountain landscape at sunset with purple clouds"
- Output: A beautiful, photorealistic image matching your description

### Why Fine-tune Stable Diffusion?

While pre-trained models like Stable Diffusion 3.5 are incredibly powerful, they might not perfectly capture:
- **Your artistic style** (e.g., watercolor paintings, anime style, vintage photographs)
- **Specific objects or characters** (your pet, a logo, architectural style)
- **Custom concepts** (fictional characters, product designs, artistic techniques)

Fine-tuning allows you to teach the model your specific visual language while maintaining its general capabilities.

## 🚀 What This Repository Does

This framework provides a **complete solution** for customizing Stable Diffusion 3.5 models:

### Key Features

✨ **Modular & Clean Code**: Well-organized, readable codebase that's easy to understand and modify  
⚙️ **Configuration-Driven**: Control everything through a single YAML file  
🔧 **Memory Efficient**: Uses 4-bit quantization and LoRA for training on consumer GPUs  
🚄 **Accelerate Integration**: Seamless multi-GPU and mixed-precision training  
📊 **Advanced Training**: Implements Rectified Flow loss for improved results  
🎯 **Easy Inference**: Simple script to generate images with your fine-tuned model  
📈 **Validation Support**: Generate test images during training to monitor progress  

### What You Can Create

- **Style Transfer**: Train on Renaissance paintings to generate images in that style
- **Character Consistency**: Create a model that generates your original character designs
- **Product Visualization**: Generate product images in various settings
- **Artistic Workflows**: Develop models for specific artistic techniques or mediums

## 📋 Requirements

### Hardware Requirements
- **Minimum**: NVIDIA GPU with 12GB VRAM (RTX 3060 12GB, RTX 4060 Ti)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (RTX 4070 Ti, RTX 4080, RTX 4090)
- **Cloud Options**: Google Colab Pro, Paperspace, RunPod, or AWS/GCP instances

### Software Requirements
- Python 3.8+
- CUDA 11.8+ or 12.x
- Git

## 🛠️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/sd3-lora-finetuning.git
cd sd3-lora-finetuning
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Accelerate (CRUCIAL)
```bash
accelerate config
```

**Example configuration for single GPU:**
```
In which compute environment are you running? → This machine
Which type of machine are you using? → No distributed training
Do you want to run your training on CPU only? → NO
Do you wish to optimize your script with torch dynamo? → NO
Do you want to use DeepSpeed? → NO
What GPU(s) should be used for training? → all (or 0)
Do you wish to use mixed precision? → bf16
```

### Step 5: Hugging Face Authentication
The Stable Diffusion 3 models require authentication:

1. Visit [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
2. Accept the license terms
3. Get your access token from [HF Settings](https://huggingface.co/settings/tokens)
4. Login via terminal:
```bash
huggingface-cli login
```

## 📁 Dataset Preparation

Your dataset should follow this structure:
```
MyDataset/
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── metadata.jsonl
```

### metadata.jsonl Format
Each line should be a JSON object:
```json
{"file_name": "image1.jpg", "text": "A detailed description of the image"}
{"file_name": "image2.png", "text": "Another descriptive caption"}
```

### Dataset Tips
- **Quality over Quantity**: 50-200 high-quality images often work better than thousands of poor ones
- **Consistent Style**: For style training, use images with consistent artistic approach
- **Detailed Captions**: Write descriptive, specific captions (10-50 words per image)
- **Image Resolution**: Use high-resolution images (1024x1024 or higher)

**For comprehensive dataset creation guidance, check out our [Dataset Preparation Toolkit](https://github.com/gokhaneraslan/text_to_image_dataset_toolkit).**

## ⚙️ Configuration

Edit `configs/training_config.yaml` to customize your training:

```yaml
# Key settings to adjust:
model:
  id: "stabilityai/stable-diffusion-3.5-medium"  # or 3.5-large
  use_quantization: true  # Set to false if you have plenty of VRAM

paths:
  output_dir: "/path/to/your/outputs"
  dataset_path: "/path/to/your/dataset"

training:
  num_epochs: 20          # Increase for more training
  batch_size: 1           # Increase if you have more VRAM
  
lora:
  rank: 32               # Higher = more parameters, better quality
  alpha: 64              # Scaling factor for LoRA
```

## 🚂 Training Your Model

### Start Training
```bash
accelerate launch train.py --config_path configs/training_config.yaml
```

### Monitor Progress
- Check console output for loss values
- Validation images saved to `output_dir/validation_images/` (if enabled)
- TensorBoard logs in `output_dir/logs/`

```bash
# View training progress with TensorBoard
tensorboard --logdir output_dir/logs
```

### Training Time Estimates
- **RTX 3060 12GB**: ~2-4 hours for 20 epochs with 100 images
- **RTX 4090**: ~30-60 minutes for 20 epochs with 100 images
- **Google Colab**: ~1-2 hours (varies with availability)

## 🎨 Generating Images

After training completes, use your fine-tuned model:

```bash
python inference.py \
    --lora_checkpoint_dir "output_dir/checkpoint-final" \
    --prompt "Your creative prompt here" \
    --output_path "my_generated_image.png" \
    --steps 28 \
    --guidance_scale 7.0
```

### Example Prompts
```bash
# For an anime-style model
python inference.py \
    --lora_checkpoint_dir "checkpoints/anime_style" \
    --prompt "anime style, a girl with blue hair in a magical forest, detailed, beautiful lighting" \
    --output_path "anime_girl.png"

# For a product visualization model
python inference.py \
    --lora_checkpoint_dir "checkpoints/product_viz" \
    --prompt "modern minimalist chair in a bright living room, professional photography" \
    --output_path "chair_visualization.png"
```

## 🐛 Common Issues & Solutions

### 1. Out of Memory (OOM) Errors
**Symptoms**: CUDA out of memory, training crashes
**Solutions**:
```yaml
# In your config file:
training:
  batch_size: 1                    # Reduce batch size
  gradient_accumulation_steps: 8   # Increase to maintain effective batch size
model:
  use_quantization: true          # Enable 4-bit quantization
```

### 2. "Model is Gated" Error
**Symptoms**: Cannot download model files
**Solutions**:
1. Accept license at [HuggingFace model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
2. Run `huggingface-cli login` with valid token
3. Ensure token has read permissions

### 3. Slow Training Speed
**Symptoms**: Very slow training progress
**Solutions**:
- Ensure CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify mixed precision is enabled in accelerate config
- Check if quantization is enabled in config
- Monitor GPU usage with `nvidia-smi`

### 4. Poor Generated Image Quality
**Symptoms**: Blurry, artifacts, doesn't match training data
**Solutions**:
- **Increase training epochs**: Try 30-50 epochs
- **Improve dataset quality**: Better captions, consistent style
- **Adjust LoRA rank**: Increase from 32 to 64 or 128
- **Lower learning rate**: Reduce all learning rates by 50%

### 5. "Accelerate Not Configured" Warning
**Symptoms**: Warning messages about accelerate configuration
**Solution**: Run `accelerate config` and follow the prompts

### 6. Dataset Loading Errors
**Symptoms**: "No valid image-caption pairs found"
**Solutions**:
- Verify `metadata.jsonl` format (each line must be valid JSON)
- Ensure image files exist in `images/` subdirectory
- Check file permissions and paths
- Validate JSON format: `python -m json.tool metadata.jsonl`

## 📊 Understanding Training Metrics

### Key Metrics to Monitor
- **Loss**: Should generally decrease over time (0.1-0.01 is typical)
- **Learning Rate**: Should follow your scheduler (cosine/linear decay)
- **Steps per Second**: Indicates training speed

### When to Stop Training
- Loss plateaus for many epochs
- Validation images show good quality
- Signs of overfitting (loss decreases but quality doesn't improve)

## 🔧 Advanced Configuration

### Memory Optimization
```yaml
# For limited VRAM
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  mixed_precision: "bf16"
model:
  use_quantization: true
```

### Quality Optimization
```yaml
# For better results (requires more VRAM)
lora:
  rank: 64        # or 128
  alpha: 128      # or 256
training:
  num_epochs: 30
optimizer:
  transformer_lr: 1.0e-6  # Lower learning rate
```

## 📁 Project Structure

```
sd3-lora-finetuning/
├── configs/
│   └── training_config.yaml    # All training parameters
├── src/
│   ├── config_loader.py        # Configuration management
│   ├── data_processing.py      # Dataset handling
│   ├── model_setup.py          # Model and LoRA setup
│   ├── trainer.py              # Training loop
│   ├── utils.py                # Utility functions
│   └── inference_during_training.py  # Validation inference
├── train.py                    # Main training script
├── inference.py                # Image generation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🤝 Contributing

We welcome contributions! Areas where help is appreciated:
- Additional model architectures
- New training techniques
- Documentation improvements
- Bug fixes and optimizations

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Diffusion 3
- [Hugging Face](https://huggingface.co/) for the transformers and diffusers libraries
- [Microsoft](https://github.com/microsoft/LoRA) for the LoRA technique
- The open-source AI community for continuous innovation

## 📚 Additional Resources

### Learning Resources
- [Understanding Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206)

### Tools & Utilities
- [Dataset Preparation Toolkit](https://github.com/gokhaneraslan/text_to_image_dataset_toolkit)
- [Image Captioning Tools](https://github.com/salesforce/BLIP)
- [Training Monitoring Dashboard](https://tensorboard.dev/)

---

**Happy Fine-tuning! 🎨✨**

*If you create something amazing with this framework, we'd love to see it! Share your results and tag us.*
