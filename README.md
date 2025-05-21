# awsome-llm-papers

A comprehensive repository for research papers, code snippets, and notes related to LLMs.


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


---

## PreTraining


###

The paper introduces the Pre-trained Model Average (PMA) strategy for merging model weights during the pre-training phase of large language models (LLMs). PMA combines checkpoints from the stable training phase using methods like Simple Moving Average (SMA), Weighted Moving Average (WMA), and Exponential Moving Average (EMA). The study shows that PMA not only enhances model performance but also enables accurate prediction of annealing behavior, leading to more efficient development and lower training costs. The optimal merging interval scales with model size, and incorporating more checkpoints improves performance.

Paper: [Model Merging
in Pre-training of Large Language Models](./papers/Model Merging
in Pre-training of Large Language Models.pdf)


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


### DPO

DPO (Direct Preference Optimization) focuses on optimizing models based on direct human preferences. This approach enhances model alignment with user intentions by incorporating preference data directly into the training process.

Paper: [DPO](./papers/DPO.pdf)


### NeurIPS-2022-training-language-models-to-follow-instructions-with-human-feedback-Paper-Conference

This paper presents methods for training language models to follow human instructions more effectively using reinforcement learning from human feedback. 

Paper: [NeurIPS-2022 Training Language Models to Follow Instructions with Human Feedback](./papers/NeurIPS-2022-training-language-models-to-follow-instructions-with-human-feedback-Paper-Conference.pdf)

---


















