## LLM-G-Code-Generator

**Code and dataset for the paper: "Large language models for G-Code generation in design for additive manufacturing"**

This repository provides an end-to-end framework for generating 3D printing G-code directly from design parameters using Large Language Models (LLMs). All scripts are located in the root directory for ease of use.

## 🚀 Usage Guide

### 1. Installation

Install the required dependencies.

```bash
pip install -r requirements.txt
```

### 2. Model Fine-tuning

Run the training script for the desired model (e.g., Qwen1-G-Coder).

```bash
python model_training_Qwen1-G-Coder.py
```

### 3. Model Evaluation

Calculate the Cross Entropy Loss and Perplexity (PPL) on the validation and test sets (e.g., Qwen1-G-Coder).

```bash
python model_evaluate_Qwen1-G-Coder.py
```

### 4. Model Inference

Generate raw G-code using the fine-tuned model (e.g., Qwen1-G-Coder).

```bash
python model_inference_Qwen1-G-Coder.py
```

### 5. Post-Processing

To generate a printable .gcode file, you must run the following three scripts in order.

#### Step 1: Add Header and Footer

Complete the G-code with machine start/end sequences.

```bash
python G-code-post-processing-complete.py
```

#### Step 2: Path Filling 

Optimize the single-layer path for closed loops.

```bash
python G-code-post-processing-single_layer_filled.py
```

#### Step 3: Multi-layer Stacking 

Create the final printable file (e.g., 20 layers).

```bash
python G-code-post-processing-multi_layers_printable.py
```


## 💾 Dataset

The dataset is located in the dataset/ directory. It contains 5,898 pairs of parametric gear designs (z, m, d) and their corresponding Ground Truth G-codes.


## 📂 Directory Structure

```text
.
├── dataset/                                # Dataset folder
│   └── stl_files/                          # Original CAD models (.stl)
│   ├── gcode_files/                        # Original slicer outputs (.gcode)
│   ├── train/                              # Training set (90%)
│   ├── validation/                         # Validation set (5%)
│   └── test/                               # Test set (5%)
│
├── requirements.txt                        # Python dependencies
├── images/                                 # Visualization plots (Loss & PPL)
│
├── [Model Fine-tuning Scripts]
├── model_training_Qwen1-G-Coder.py         # Fine-tuning script for Qwen1-G-Coder
├── model_training_Qwen2-G-Coder.py         # Fine-tuning script for Qwen2-G-Coder
├── model_training_DeepSeek-G-Coder.py      # Fine-tuning script for DeepSeek-G-Coder
│
├── [Model Inference Scripts]
├── model_inference_Qwen1-G-Coder.py        # Inference script for Qwen1-G-Coder
├── model_inference_Qwen2-G-Coder.py        # Inference script for Qwen2-G-Coder
├── model_inference_DeepSeek-G-Coder.py     # Inference script for DeepSeek-G-Coder
│
├── [Model Evaluation Scripts]
├── model_evaluate_Qwen1-G-Coder.py         # Evaluation script for Qwen1-G-Coder
├── model_evaluate_Qwen2-G-Coder.py         # Evaluation script for Qwen2-G-Coder
├── model_evaluate_DeepSeek-G-Coder.py      # Evaluation script for DeepSeek-G-Coder
│
├── [Post-Processing Scripts] (Run in order)
├── G-code-post-processing-complete.py               # Step 1: Add Header/Footer (Single Layer Unfilled)
├── G-code-post-processing-single_layer_filled.py    # Step 2: Path Filling (Single Layer)
└── G-code-post-processing-multi_layers_printable.py # Step 3: Multi-layer Stacking (Final Output Printable)
│
├── [Results]
├── Qwen1-G-Coder_generate_gcode_files/                 # Qwen1-G-Coder Raw LLM outputs (Core toolpath only)
├── Qwen1-G-Coder_generate_gcode_files_printable/       # Qwen1-G-Coder Final printable G-codes
├── Qwen2-G-Coder_generate_gcode_files/                 # Qwen2-G-Coder Raw LLM outputs (Core toolpath only)
├── Qwen2-G-Coder_generate_gcode_files_printable/       # Qwen2-G-Coder Final printable G-codes
├── DeepSeek-G-Coder_generate_gcode_files/              # DeepSeek-G-Coder Raw LLM outputs (Core toolpath only)
├── DeepSeek-G-Coder_generate_gcode_files_printable/    # DeepSeek-G-Coder Final printable G-codes

 ```
```markdown

## 📊 Results & Visualization

### 1. Training Loss Comparison
![Training Loss](images/Training_Loss.png)
*Figure 1: Comparison of training loss convergence across three LLMs.*

### 2. Loss & PPL Analysis
![Loss and PPL](images/Model_Evaluation_Metrics.png)
*Figure 2: Loss and Perplexity (PPL) metrics of the dataset.*   



