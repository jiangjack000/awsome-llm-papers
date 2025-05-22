# awesome-llm-papers

A comprehensive repository for research papers, code snippets, and notes related to LLMs.

## Summary
- [awesome-llm-papers](#awesome-llm-papers)
  - [Summary](#summary)
  - [OutStanding LLM Technical Report](#outstanding-llm-technical-report)
    - [GPT-4](#gpt-4)
    - [GPT-4.5](#gpt-45)
    - [Qwen2\_5\_1M\_Technical\_Report](#qwen2_5_1m_technical_report)
    - [KIMI-1.5](#kimi-15)
    - [DeepSeek\_R1](#deepseek_r1)
    - [DeepSeek\_V3](#deepseek_v3)
    - [LLama3.1](#llama31)
  - [Architecture](#architecture)
    - [Attention Is All You Need](#attention-is-all-you-need)
    - [ERNIE - Enhanced Language Representation with Informative Entities](#ernie---enhanced-language-representation-with-informative-entities)
    - [XLNet - Generalized Autoregressive Pretraining for Language Understanding](#xlnet---generalized-autoregressive-pretraining-for-language-understanding)
    - [RoBERTa - A Robustly Optimized BERT Pretraining Approach](#roberta---a-robustly-optimized-bert-pretraining-approach)
    - [Swin Transformer - Hierarchical Vision Transformer using Shifted Windows](#swin-transformer---hierarchical-vision-transformer-using-shifted-windows)
    - [Learning Transferable Visual Models From Natural Language Supervision](#learning-transferable-visual-models-from-natural-language-supervision)
  - [Parallel Training](#parallel-training)
    - [Training\_Multi-Billion\_Parameter\_Language\_Models\_Usin\_Model\_Parallelism](#training_multi-billion_parameter_language_models_usin_model_parallelism)
    - [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](#gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism)
    - [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](#zero-memory-optimizations-toward-training-trillion-parameter-models)
    - [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](#deepspeed-system-optimizations-enable-training-deep-learning-models-with-over-100-billion-parameters)
    - [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](#switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-sparsity)
  - [PreTraining](#pretraining)
    - [Model Merging in Pre-training of Large Language Models](#model-merging-in-pre-training-of-large-language-models)
    - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](#exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer)
    - [Improving Language Understanding by Generative Pre-Training](#improving-language-understanding-by-generative-pre-training)
    - [BERT](#bert)
    - [Language Models are Few-Shot Learners](#language-models-are-few-shot-learners)
    - [OLMO2](#olmo2)
  - [Post-Training](#post-training)
    - [LEARNING DYNAMICS OF LLM FINETUNING  ICLR 2025 Oral, Outstanding Paper Award](#learning-dynamics-of-llm-finetuning--iclr-2025-oral-outstanding-paper-award)
    - [DPO](#dpo)
    - [NeurIPS-2022-training-language-models-to-follow-instructions-with-human-feedback-Paper-Conference](#neurips-2022-training-language-models-to-follow-instructions-with-human-feedback-paper-conference)


## OutStanding LLM Technical Report
### GPT-4
GPT-4 is OpenAI's large multimodal language model, capable of processing text and visual inputs.

Paper: [GPT-4](./papers/GPT-4_Technical_Report.pdf)

### GPT-4.5
GPT-4.5, is an advanced version of GPT-4 with enhanced pattern recognition and creativity. It supports 15 languages and offers improved conversational abilities.

Paper: [GPT-4.5](./papers/gpt-4-5-system-card-2272025.pdf)



### Qwen2_5_1M_Technical_Report

The Qwen2.5-1M models extend context length to 1 million tokens. The series includes open-source models Qwen2.5-7B-Instruct-1M and Qwen2.5-14B-Instruct-1M, and the API-accessible Qwen2.5-Turbo.

Paper: [Qwen2_5_1M_Technical_Report](./papers/Qwen2_5_1M_Technical_Report.pdf)

### KIMI-1.5

### DeepSeek_R1

DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) .

Paper: [DeepSeek-R1](./papers/DeepSeek_R1.pdf)


### DeepSeek_V3

DeepSeek_V3 introduces enhancements over its predecessors by incorporating advanced optimization techniques and expanded training data.

Paper: [DeepSeek_V3](./papers/DeepSeek_V3.pdf)

### LLama3.1

The largest model is a dense Transformer with 405B parameters, processing information in a context window of up to 128K tokens.

Paper: [LLama3.1](./papers/llama3.1.pdf)

---



## Architecture

### Attention Is All You Need

The "Attention Is All You Need" paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms.

Paper: [Attention Is All You Need](./papers/Attention_Is_All_You_Need.pdf)


### ERNIE - Enhanced Language Representation with Informative Entities

ERNIE enhances language representation by integrating structured knowledge about entities.

Paper: [ERNIE - Enhanced Language Representation with Informative Entities](./papers/ERNIE_Enhanced_Language_Representation_with_Informative_Entities.pdf)



### XLNet - Generalized Autoregressive Pretraining for Language Understanding

XLNet combines autoregressive modeling with autoencoding to capture bidirectional context without the limitations of BERT. 

Paper: [XLNet - Generalized Autoregressive Pretraining for Language Understanding](./papers/XLNet_Generalized_Autoregressive_Pretraining_for_Language_Understanding.pdf)



### RoBERTa - A Robustly Optimized BERT Pretraining Approach

RoBERTa optimizes the BERT pretraining process by removing the next sentence prediction objective and training with larger mini-batches and learning rates. 

Paper: [RoBERTa - A Robustly Optimized BERT Pretraining Approach](./papers/RoBERTa_A_Robustly_Optimized_BERT_Pretraining_Approach.pdf)

---

### Swin Transformer - Hierarchical Vision Transformer using Shifted Windows

The Swin Transformer introduces a hierarchical structure using shifted windows for attention computation. 

Paper: [Swin Transformer - Hierarchical Vision Transformer using Shifted Windows](./papers/Swin_Transformer_Hierarchical_Vision_Transformer_using_Shifted_Windows.pdf)

---

### Learning Transferable Visual Models From Natural Language Supervision

The CLIP model learns visual representations by training on image-text pairs from the internet. 

Paper: [Learning Transferable Visual Models From Natural Language Supervision](./papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf)

---




## Parallel Training

### Training_Multi-Billion_Parameter_Language_Models_Usin_Model_Parallelism

This paper presents techniques for efficiently training language models with billions of parameters via model parallelism, enabling the scaling of model size beyond single GPU memory limitations.

Paper: [Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053)



### GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

GPipe proposes a pipeline parallelism approach that splits models into segments and processes micro-batches in a pipeline, significantly improving memory efficiency and scalability.

Paper: [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/pdf/1811.06965)



### ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

ZeRO (Zero Redundancy Optimizer) introduces a set of memory optimization techniques that enable the training of trillion-parameter models by partitioning optimizer states, gradients, and parameters across devices.

Paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054)



### DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters

DeepSpeed presents a deep learning optimization library that combines ZeRO and other system techniques to enable efficient training of models with over 100 billion parameters.

Paper: [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://arxiv.org/pdf/2007.03029)



### Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

Switch Transformers leverages sparse expert models and routing to enable the scaling of language models to over a trillion parameters while maintaining computational efficiency.

Paper: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961)


---

## PreTraining


### Model Merging in Pre-training of Large Language Models

The paper introduces the Pre-trained Model Average (PMA) strategy for merging model weights during the pre-training phase of large language models (LLMs). 

Paper: [Model Merging in Pre-training of Large Language Models](./papers/Model_Merging_in_Pre-training_of_Large_Language_Models.pdf)


### Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

This paper explores the capabilities of a unified text-to-text framework for transfer learning.

Paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](./papers/Exploring_the_Limits_of_Transfer_Learning_with_a_Unified_Text-to-Text_Transformer.pdf)


### Improving Language Understanding by Generative Pre-Training

The Generative Pre-Training (GPT) model improves language understanding by pre-training on a large corpus of text data and then fine-tuning on specific tasks. 

Paper: [Improving Language Understanding by Generative Pre-Training](./papers/Improving_Language_Understanding_by_Generative_Pre-Training.pdf)

### BERT

BERT (Bidirectional Encoder Representations from Transformers) presents a novel approach to pre-training language representations.

Paper: [BERT](./papers/BERT.pdf)


### Language Models are Few-Shot Learners

This paper demonstrates that large language models, like GPT-3, can perform tasks with little to no task-specific training.

Paper: [Language Models are Few-Shot Learners](./papers/Language_Models_are_Few-Shot_Learners.pdf)


### OLMO2

Pre Train 2 OLMO2 discusses the advancements in pretraining strategies for the OLMO2 model.
Paper: [pre_train_2_OLMO2](./papers/pre_train_2_OLMO2.pdf)

---





## Post-Training

### LEARNING DYNAMICS OF LLM FINETUNING  ICLR 2025 Oral, Outstanding Paper Award


It provides a framework to analyze how the learning of specific training examples influences the model's predictions on other examples, offering insights into the behavior of deep learning systems.

Paper: [DPO](./papers/LEARNING_DYNAMICS_OF_LLM_FINETUNING.pdf)

### DPO

DPO (Direct Preference Optimization) focuses on optimizing models based on direct human preferences. This approach enhances model alignment with user intentions by incorporating preference data directly into the training process.

Paper: [DPO](./papers/DPO.pdf)


### NeurIPS-2022-training-language-models-to-follow-instructions-with-human-feedback-Paper-Conference

This paper presents methods for training language models to follow human instructions more effectively using reinforcement learning from human feedback. 

Paper: [NeurIPS-2022 Training Language Models to Follow Instructions with Human Feedback](./papers/NeurIPS-2022-training-language-models-to-follow-instructions-with-human-feedback-Paper-Conference.pdf)

---


















