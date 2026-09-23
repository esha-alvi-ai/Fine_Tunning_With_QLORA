# QLoRA Fine-Tuning
## Overview

**QLoRA (Quantized Low-Rank Adaptation)** combines:

1.  Low-Rank Adaptation (LoRA)
2.  4-bit quantization
3.  Parameter-efficient fine-tuning

The base LLM is loaded in low precision, usually 4-bit, kept frozen, and
adapted using trainable LoRA matrices.

``` text
                 QLoRA
                   |
          +--------+--------+
          |                 |
     Quantization          LoRA
          |                 |
       4-bit            A + B matrices
          |                 |
    Frozen Base          Trainable
          |                 |
          +--------+--------+
                   |
             Fine-Tuned LLM
```

------------------------------------------------------------------------

## Why QLoRA?

Full fine-tuning can require very large GPU memory.

LoRA reduces trainable parameters, but the full-precision base model
still occupies substantial memory.

QLoRA additionally quantizes the base model.

``` text
Full Fine-Tuning
    ↓
Full-precision base + all weights train
    ↓
Very high memory

LoRA
    ↓
Normal base + small LoRA adapters
    ↓
Lower memory

QLoRA
    ↓
4-bit base + small LoRA adapters
    ↓
Lower base-model memory
```

------------------------------------------------------------------------

## QLoRA Architecture

``` mermaid
flowchart TD
    A[Pretrained LLM] --> B[4-bit Quantization]
    B --> C[Frozen Quantized Base Model]
    C --> D[Transformer Layer]

    D --> E[Base Weight Wq]
    D --> F[LoRA Branch]

    F --> G[Matrix A]
    G --> H[Matrix B]
    H --> I[Alpha / Rank Scaling]

    E --> J[Base Output]
    I --> K[LoRA Update]

    J --> L[Add]
    K --> L
    L --> M[Next Layer]

    N[Training Data] --> O[Tokenizer]
    O --> M

    M --> P[Loss]
    P --> Q[Backpropagation]
    Q --> R[Update LoRA Only]
```

------------------------------------------------------------------------

## Core Equation

LoRA uses:

\[ W' = W + `\frac{\alpha}{r}`{=tex}BA \]

In QLoRA, the base weights are quantized:

\[ W_q `\approx `{=tex}Q(W) \]

Conceptually:

\[ Y = XW_q + X`\frac{\alpha}{r}`{=tex}BA \]

The quantized base is kept frozen while `A` and `B` are trained.

------------------------------------------------------------------------

## 4-Bit Quantization

Typical model storage precision:

``` text
FP32 → 32 bits
FP16 → 16 bits
BF16 → 16 bits
INT8 → 8 bits
4-bit → 4 bits
```

QLoRA commonly uses **NF4 (NormalFloat 4-bit)**.

Important:

> 4-bit storage does not mean every computation happens in 4-bit.

A typical setup stores the base model in 4-bit and performs computation
using FP16 or BF16.

------------------------------------------------------------------------

## NF4

NF4 is a 4-bit data type designed for quantized neural-network weights.

Typical configuration:

``` python
bnb_4bit_quant_type="nf4"
```

------------------------------------------------------------------------

## Double Quantization

QLoRA can quantize not only model weights but also quantization
constants.

``` python
bnb_4bit_use_double_quant=True
```

Conceptually:

``` text
Weights
   ↓
4-bit quantization
   ↓
Quantization constants
   ↓
Additional quantization
```

This reduces memory overhead.

------------------------------------------------------------------------

## BitsAndBytes Configuration

``` python
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

### Meaning

``` text
load_in_4bit=True
    → load base model using 4-bit quantization

bnb_4bit_quant_type="nf4"
    → use NF4

bnb_4bit_use_double_quant=True
    → enable double quantization

bnb_4bit_compute_dtype=torch.float16
    → use FP16 for computation
```

------------------------------------------------------------------------

## QLoRA Training Architecture

``` mermaid
flowchart LR
    A[Instruction Dataset] --> B[Tokenizer]
    B --> C[Token IDs]

    C --> D[4-bit Quantized Base LLM]
    D --> E[Forward Pass]
    E --> F[Logits]
    F --> G[Loss]
    G --> H[Backward Pass]
    H --> I[LoRA Gradients]
    I --> J[Update A and B]

    D -. Frozen .-> K[Base Parameters]
    K -. No optimizer update .-> K
```

------------------------------------------------------------------------

## Complete QLoRA Code

### Install

``` bash
pip install -U transformers datasets peft accelerate bitsandbytes
```

### Imports

``` python
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
```

### Model

``` python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
```

### Quantization

``` python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

### Load model

``` python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Prepare for k-bit training

``` python
model = prepare_model_for_kbit_training(model)
```

### LoRA configuration

``` python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Attach LoRA

``` python
model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

------------------------------------------------------------------------

## QLoRA Dataset Flow

``` text
Raw Dataset
     ↓
Instruction + Response
     ↓
Chat/Prompt Formatting
     ↓
Tokenizer
     ↓
input_ids + attention_mask
     ↓
Data Collator
     ↓
Training Batch
```

A basic educational format:

``` text
### Instruction:
What is QLoRA?

### Response:
QLoRA combines 4-bit quantization with LoRA.
```

For serious instruction tuning, use the base model's recommended chat
template and consider assistant-only loss.

------------------------------------------------------------------------

## Training

``` python
training_args = TrainingArguments(
    output_dir="./qlora_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
    remove_unused_columns=False
)
```

Then:

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator
)

trainer.train()
```

------------------------------------------------------------------------

## What Gets Updated?

``` text
Base model:
    4-bit
    Frozen
    NOT directly optimized

LoRA:
    Trainable
    Updated by optimizer
```

This is the core QLoRA idea.

------------------------------------------------------------------------

## Saving the Adapter

``` python
model.save_pretrained("./my_qlora_adapter")
tokenizer.save_pretrained("./my_qlora_adapter")
```

The adapter is much smaller than saving a separate full copy of the base
model.

------------------------------------------------------------------------

## Loading the Adapter

``` python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

trained_model = PeftModel.from_pretrained(
    base_model,
    "./my_qlora_adapter"
)
```

Architecture:

``` text
4-bit Base Model
       +
QLoRA Adapter
       ↓
Fine-Tuned Model
```

------------------------------------------------------------------------

## Important QLoRA Hyperparameters

  Parameter                       Purpose
  ------------------------------- -------------------------------------
  `load_in_4bit`                  Enables 4-bit loading
  `nf4`                           4-bit quantization type
  `double_quant`                  Additional quantization
  `compute_dtype`                 Computation precision
  `r`                             LoRA rank
  `lora_alpha`                    LoRA scaling
  `lora_dropout`                  Regularization
  `target_modules`                Layers receiving LoRA
  `learning_rate`                 Training update size
  `batch_size`                    Examples per device
  `gradient_accumulation_steps`   Accumulates gradients before update

------------------------------------------------------------------------

## QLoRA vs LoRA

  Feature          LoRA                 QLoRA
  ---------------- -------------------- ------------------------------
  Base model       Standard precision   4-bit quantized
  Base weights     Frozen               Frozen
  LoRA adapters    Trainable            Trainable
  Memory           Lower                Usually lower
  Main technique   PEFT                 Quantization + PEFT
  Typical use      Fine-tuning          Memory-efficient fine-tuning

------------------------------------------------------------------------

## QLoRA vs Full Fine-Tuning

``` text
FULL FINE-TUNING
Model
 ↓
All weights train
 ↓
Very high memory

QLoRA
Model
 ↓
4-bit quantized
 ↓
Frozen
 +
LoRA
 ↓
Small trainable adapters
```

------------------------------------------------------------------------

## QLoRA vs RAG

QLoRA changes model behavior through training.

RAG supplies external information at inference time.

``` text
QLoRA:
Base Model → Fine-Tuning → Adapted Model

RAG:
Question → Retrieval → Documents → LLM → Answer
```

They can be combined:

``` text
                 User
                   |
                   v
              QLoRA LLM
                   |
             +-----+-----+
             |           |
          Behavior      RAG
             |           |
             |       Product Data
             |           |
             +-----+-----+
                   |
                Answer
```

------------------------------------------------------------------------

## When QLoRA Is Useful

QLoRA is useful when:

-   GPU memory is limited.
-   You want to fine-tune a larger model.
-   You want task-specific behavior.
-   You want a small adapter file.
-   You need multiple task-specific adapters.
-   You want to experiment with instruction tuning.

------------------------------------------------------------------------

## Limitations

-   Quantization can introduce some approximation error.
-   Very small datasets can cause overfitting.
-   Adapter quality depends strongly on training data.
-   Target module names vary by model.
-   Not every model supports every quantization configuration.
-   Free Colab GPU availability and memory can vary.
-   A small educational dataset should not be treated as a production
    training dataset.

------------------------------------------------------------------------

## Key Interview Definition

> QLoRA is a parameter-efficient fine-tuning method that loads a
> pretrained model in low-bit precision, commonly 4-bit, freezes the
> quantized base model, and trains LoRA adapters on selected layers.

------------------------------------------------------------------------

## Remember This

``` text
QLoRA

Q = Quantization
LoRA = Low-Rank Adaptation

QLoRA = 4-bit Quantized Base Model + LoRA Adapter
```
