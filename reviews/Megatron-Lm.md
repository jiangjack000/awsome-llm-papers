


### 什么是AI系统
- AI系统是AI的体系结构，包括硬件和软件。从下到上主要包括，硬件，编译架构，训练和推理架构，Python和上层算法


### 为什么你使用Megatron，不使用DeepSpeed？
### 除了Megatron，你还了解哪些训练框架？

### 反向是前向的两倍？


#### 1. 以线性层（全连接层）为例

假设有如下前向传播公式：
$$
y = Wx + b
$$
其中，$x \in \mathbb{R}^n$，$y \in \mathbb{R}^m$，$W \in \mathbb{R}^{m \times n}$。

- **前向传播计算量：**
  - 主要是一次矩阵乘法 $Wx$，计算量为 $O(mn)$。

- **反向传播计算量：**
  - 需要计算损失对参数和输入的梯度，包括：
    1. 对权重的梯度：
       $$
       \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T
       $$
       计算量 $O(mn)$。
    2. 对输入的梯度：
       $$
       \frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial y}
       $$
       计算量 $O(mn)$。
    3. 对偏置的梯度：
       $$
       \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}
       $$
       计算量可忽略。

- **合计：**
  - 前向传播：$O(mn)$
  - 反向传播：$O(mn) + O(mn) = 2O(mn)$



**反向传播的计算量通常是前向传播的2倍**，原因是反向传播需要为每个参数和输入分别计算梯度，每一项都需要一次与前向传播类似的计算。

---
## DeepSpeed

### 混合精度训练
### 训练8B模型需要多少显存

###  ZERO
- ZeRO通过消除分布式训练中的冗余存储，将模型状态（参数、梯度、优化器状态）分片到多个设备，实现显存占用的线性降低。
- 首先先计算一下训练过程中的内存需求，模型训练包括模型参数，梯度，Adam优化器beta和gamma，以及激活值。
  - 假设采用混合精度训练，梯度和参数使用bf16，Adam两个参数采用float32，保存一份float32的参数数值。
  - 假设模型大小为a
  - 梯度和参数是bf16，总共为4a
  - Adam两个参数的是float32，总共为8a
  - 保存一份float32的参数数值，为4a
  - 激活值和batch_size和模型结构有关：为batch_size * 序列长度 * hidden_size * 层的数量 * 2字节
  - 最后还有一些临时缓冲区，包括梯度累积、通信缓冲区等。
- Zero1 将优化器状态（如Adam的动量、方差）分片到各GPU，假设有N张卡，则每张卡的优化器状态变为原来的 1 /  N，参数和梯度保持完整副本。他的通信开销：仅需在参数更新时同步梯度（与标准数据并行相同）
- Zero2 将梯度分片存储，每个GPU仅保留部分梯度，显存得到进一步的节省。提升了通信开销，：需All-Gather操作同步梯度
- Zero3 将参数分片存储，每个GPU仅保留部分参数，此时需要大量的All-Gather。


### 8B模型推理
- 模型训练包括模型参数，中间激活值和其他开销。
- 假设模型大小是a，使用bf16来存储，这就是2a字节
- 激活值和batch_size和模型结构有关：为batch_size * 序列长度 * hidden_size * 层的数量 * 2字节
- 还有些额外开销，比如用于CUDA kernel等额外开销。



### Gpipe
## 8B模型训练
- 模型训练包括模型参数，梯度，Adam优化器beta和gamma，以及激活值。
- 假设采用混合精度训练，梯度和参数使用bf16，Adam两个参数采用float32，保存一份float32的参数数值。
- 假设模型大小为a
- 梯度和参数是bf16，总共为4a
- Adam两个参数的是float32，总共为8a
- 保存一份float32的参数数值，为4a
- 激活值和batch_size和模型结构有关：为batch_size * 序列长度 * hidden_size * 层的数量 * 2字节
- 最后还有一些临时缓冲区，包括梯度累积、通信缓冲区等。
- 总结一下就是16 * a + 激活值 + 临时缓冲区。训练一个8B的模型，大概需要100G左右。


## KV-cache
## 数据并行，张量并行，模型并行，层内并行和层外并行




### 激活检查点（Activation Checkpointing）
- 在传统训练模式中前向传播：保存所有中间激活值但显存占用高，在反向传播时候：直接使用保存的激活值计算梯度。
- 为了减少激活值从而减少整个内存占用，采用激活检查点方式，它采用了分段策略，将网络划分为多个段（Segment）
- 仅保存每个段起点的激活值反向计算，从检查点重新计算段内激活值，用计算时间换取显存空间。
- 在进行分段的时候要避免特别小的分段，确保每个分段的计算时间相近。
- 显存节省的比例为 1 - 段数 / 总层数。一般5-10层为一段。

## 5. 梯度累积（Gradient Accumulation）
- 常规训练中，每个批次计算梯度后立即更新参数。梯度累积则是累积多个小批次的梯度，最后更新一次。
- 这样，有效批次大小等于小批次大小乘以累积步数，但显存占用仅相当于单个小批次。

``` python

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
accum_steps = 4  # 累积步数

for epoch in range(epochs):
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 梯度累积，不立即清零
        
        if (i+1) % accum_steps == 0:
            optimizer.step()     # 累积足够步数后更新参数
            optimizer.zero_grad()  # 清零梯度
            
            # 可选：梯度缩放（混合精度训练时）
            # scaler.step(optimizer)
            # scaler.update()

```
### 3. 混合精度训练（Mixed Precision Training）
```
sequenceDiagram
    participant GPU
    participant CPU
    Note over GPU: 前向传播
    GPU->>GPU: 输入数据(FP32 → FP16)
    GPU->>GPU: 计算激活值(FP16)
    Note over GPU: 反向传播
    GPU->>GPU: 计算梯度(FP16)
    GPU->>CPU: 梯度转换(FP16 → FP32)
    CPU->>CPU: 参数更新(FP32)
    CPU->>GPU: 更新后参数(FP32 → FP16)

```
- 在进行训练的时候，我们一般会使用float16计算参数和激活值以及梯度，但会额外存储一份float32的模型参数
- 优化器的参数都会使用32位
| 组件         | 存储精度 | 计算精度 | 原因                     |
|--------------|----------|----------|-------------------------|
| 模型参数     | FP32     | FP16     | 避免舍入误差累积         |
| 前向激活值   | FP16     | FP16     | 利用Tensor Core加速      |
| 梯度计算     | FP16     | FP16     | 保持计算一致性           |
| 优化器状态   | FP32     | FP32     | 确保参数更新精度         |




### 损失缩放（Loss Scaling）

- 一般在模型训练时参数和梯度都是bf16，小梯度值会因下溢变成0
- 前向计算损失并放大，反向传播得到放大后的梯度，同时检查梯度是否存在Inf/NaN
  - 若无异常：保持或增大缩放因子
  - 若检测异常：跳过本次更新并减小缩放因子

|初始缩放因子 |65536 | 足够大的初始值避免早期下溢| 
|--------------|----------|----------|
|增长因子|2.0|每次成功更新后放大缩放因子|
|回退因子|0.5|检测到Inf/NaN时缩小缩放因子|

``` python
scaler = GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5)

# 训练循环中
scaled_loss = scaler.scale(loss)  # 放大损失
scaled_loss.backward()            # 反向传播放大后的梯度

if scaler.update():  # 自动调整缩放因子
    optimizer.step()
    optimizer.zero_grad()

```


### 精度转换器（Precision Converter）
- 使用Autocast实现智能选择算子计算精度,APM

``` python 

with torch.cuda.amp.autocast(dtype=torch.float16):  # 启用自动转换
    outputs = model(inputs)  # 自动选择FP16/FP32
    loss = loss_fn(outputs, targets)
```

| 运算类型                | 推荐精度 | 原因说明                          |
|-------------------------|----------|-----------------------------------|
| 矩阵乘法                | FP16     | 利用 Tensor Core 加速计算         |
| 卷积                    | FP16     | 利用 Tensor Core 加速计算         |
| Softmax                 | FP32     | 指数运算对高精度要求较高          |
| 层归一化                | FP32     | 方差计算易出现数值不稳定          |
| 小规模逐元素操作        | FP32     | 避免累积误差影响最终结果          |


### 梯度处理器（Gradient Handler）
- 梯度裁剪（Gradient Clipping），放大后的梯度可能引发爆炸
- 异常检测机制：Inf/NaN过滤

### 数据并行，模型并行，张量并行

- 数据并行：每个GPU保存完整的模型副本，但处理不同的数据子集。训练过程中，每个GPU独立计算其数据子集的梯度，然后通过全局通信操作（AllReduce）将梯度汇总并更新模型参数。数据并行适合处理大规模数据集，减少每个GPU的计算量。但也有一些问题
  - 并行度太高的话，每个GPU运行的batch-size变得太小，这降低了GPU的利用率，还增加了通信成本；
  - 可使用的最大设备数就是batch-size，这限制了可用于训练的GPU数量。
- 模型并行：在模型并行中，模型的不同部分（如层）被分配到不同的GPU上。每个GPU只保存模型的一部分，通过通信传递中间结果。模型并行适合处理超大规模模型，减少每个GPU的内存需求。
  - 模型并行可以分为流水线并行和张量并行
  - 张量并行是将模型的单层内部的参数和计算任务拆分到不同的设备上执行。具体来说，它会将张量（如权重矩阵或输入矩阵）按行或按列切分，然后在不同设备上并行执行部分计算，最后通过集合通信操作（如 AllReduce）合并结果。比如Megatron
  - 流水线并行是将模型的不同层分配到不同的设备上，每个设备负责计算模型的一部分。例如，将一个 16 层的 Transformer 模型的每一层分别放在不同的 GPU 上，每个 GPU 负责计算一层的前向和反向传播。
  - 流水线并行的实现方式
    - 层间并行：将模型的不同层分配到不同的设备上，每个设备负责计算其分配到的层。
    - 微批次（Micro-batch）：为了减少设备的空闲时间，通常会将一个大批次数据切分成多个小批次（微批次），通过流水线调度算法减少计算空泡。

### Megatron-lm
- 这是英伟达在2019年提出的一项技术，他使用模型并行技术来训练拥有数十亿参数的大语言模型，特别是基于Transformer架构的模型。文章提出了一个简单高效的模型并行方法，使得在PyTorch框架内无需额外编译器或库修改即可训练数十亿参数的Transformer模型，并在多个自然语言处理（NLP）任务中展示了其有效性。
- 他利用Transformer模型的内在结构，通过在层内（intra-layer）进行模型并行化来训练大型模型。这种方法不需要新的编译器或库修改，与流水线模型并行化（pipeline model parallelism）正交且互补，可以通过在PyTorch中插入少量通信操作来完全实现。
- MLP块的并行化：Transformer层中的MLP块包含两个GEMM（General Matrix Multiply）操作，文章中提出将第一个GEMM的权重矩阵A沿列方向分割，第二个GEMM沿行方向分割，从而避免了在非线性函数GeLU之前的同步点。
- 自注意力块的并行化：利用多头注意力操作的固有并行性，将与键（K）、查询（Q）和值（V）相关的GEMM操作以列方向分割，使得每个注意力头的矩阵乘法可以在一个GPU上本地完成，无需立即通信。
- 输出嵌入的并行化：由于现代语言模型的词汇表大小通常在数万级别，文章提出将输出嵌入GEMM操作并行化，通过将输入嵌入权重矩阵沿词汇表维度分割，并在输出嵌入后执行all-reduce操作来同步结果。
- 他在Bert和GPT上都做了实验，并在多个NLP任务上取得了最先进的结果。


## Megatron的张量并行计算在哪些地方，具体如何做？
Megatron-LM 的张量并行主要在 Transformer 模型的多层感知机（MLP）层和多头注意力（MHA）层。

### MLP 层的张量并行
在 MLP 层中，张量并行通过以下方式实现：
- **权重矩阵分割**：对于第一个线性层的权重矩阵 \( A \)，采用“列切割”，将 \( A \) 分割为多个子矩阵，每个子矩阵分配到不同的 GPU 上。例如，如果 \( A \) 是一个 \( m * n \) 的矩阵，可以将其分割为 \( A_1, A_2, 到, A_k \)，每个子矩阵分配到一个 GPU 上。
- **输入数据分配**：输入数据 \( X \) 被复制到每个 GPU 上，每个 GPU 独立计算其对应的子矩阵与输入数据的矩阵乘法。
- **非线性激活函数处理**：由于 GELU 等非线性激活函数的性质，每个 GPU 可以独立计算其子矩阵的激活结果，无需在激活前进行通信。
- **输出合并**：在第一个线性层的输出计算完成后，通过 **AllReduce** 操作将所有 GPU 上的计算结果相加，得到完整的输出。

对于第二个线性层，其权重矩阵 \( B \) 采用“行切割”，输入数据为第一个线性层的输出，直接在每个 GPU 上进行计算，无需通信。

### MHA 层的张量并行
- MHA可以分为两个部分，注意力计算和MLP
- **QKV 矩阵分割**：对于查询（Q）、键（K）和值（V）矩阵，采用“列切割”，将每个矩阵分割为多个子矩阵，分配到不同的 GPU 上。
- **多头注意力计算**：每个注意力头的计算是独立的，可以分配到不同的 GPU 上并行执行。例如，如果有 8 个注意力头，可以将每个头的计算分配到一个 GPU 上，或者多个头分配到一个 GPU 上，具体取决于 GPU 的数量和模型的设计。
- **线性层分割**：对于 MHA 层的输出线性层，其权重矩阵采用“行切割”，输入数据为多头注意力的输出，直接在每个 GPU 上进行计算。

### 嵌入层的张量并行
- 输入输出的Embedding维度通常也很大，这边也可以张量并行
- **输入嵌入层**：将嵌入矩阵按照词汇表维度进行分割，每个 GPU 存储部分嵌入向量。在前向传播时，通过 **AllGather** 操作将所有 GPU 上的部分嵌入向量汇总，得到完整的嵌入向量。
- **输出嵌入层**：由于输出嵌入层通常与输入嵌入层共享权重，因此也需要进行类似的分割和通信操作。

### 通信操作
- **AllReduce**：用于在多个 GPU 之间同步和汇总计算结果。
- **AllGather**：用于汇总所有 GPU 上的部分结果。
- **ReduceScatter**：用于将汇总后的结果分配到各个 GPU 上。

- 这些通信操作确保了在并行计算过程中，各个 GPU 上的计算结果能够正确地合并和同步。

### 初始化和配置
在 Megatron-LM 中，可以通过以下代码初始化张量并行、流水线并行和数据并行：
```python
from megatron.core import mpu, tensor_parallel

mpu.initialize_model_parallel(args.tensor_model_parallel_size,
                              args.pipeline_model_parallel_size,
                              args.virtual_pipeline_model_parallel_size,
                              args.pipeline_model_parallel_split_rank)
```
这段代码用于设置并行组的大小和配置。

通过上述方法，Megatron-LM 的张量并行计算能够有效地将大型 Transformer 模型的计算任务分配到多个 GPU 上，从而实现高效的大规模模型训练。
## Deepspeed和Megatron之间的差异是什么？

### 1. **开发背景和目标**
- **DeepSpeed**：
  - 由微软开发，旨在提供高性能、易用的分布式训练框架。
  - 提供丰富的优化功能，支持多种深度学习框架（如 PyTorch、TensorFlow）。
- **Megatron-LM**：
  - 由 NVIDIA 开发，专注于大规模 Transformer 模型的训练。
  - 主要支持 PyTorch，对 NVIDIA GPU 进行了深度优化。

### 2. **并行策略**
- **DeepSpeed**：
  - **数据并行**：通过 Zero 系列优化（Zero-1、Zero-2、Zero-3）显著降低内存占用，支持大规模数据并行。
  - **模型并行**：支持与 Megatron-LM 的集成，但自身模型并行功能相对较少。
- **Megatron-LM**：
  - **张量并行**：将模型参数和激活张量切片到多个 GPU 上，降低单个 GPU 的内存需求。
  - **流水线并行**：将模型的不同层分配到不同 GPU 上，实现高效的并行计算。

### 3. **优化功能**
- **DeepSpeed**：
  - 提供混合精度训练（FP16 和 FP32）、梯度累积、激活检查点等多种优化功能。
  - Zero 系列优化器通过减少优化器状态、梯度和参数的冗余存储，显著降低内存占用。
- **Megatron-LM**：
  - 支持混合精度训练，减少内存消耗并提高计算性能。
  - 提供灵活的模型并行策略，可以根据需求进行调整。

### 4. **易用性和集成性**
- **DeepSpeed**：
  - 提供简化的 API，只需对 PyTorch 模型进行少量代码更改即可启用。
  - 支持多个深度学习框架，便于与现有系统集成。
- **Megatron-LM**：
  - 主要支持 PyTorch，对其他框架的支持较为有限。
  - 提供详细的文档和示例代码，便于用户理解和使用。

### 5. **性能表现**
- **DeepSpeed**：
  - 在数据并行方面表现出色，特别是在内存优化和训练速度方面。
  - 通过 Zero 系列优化，能够在有限的资源下训练更大的模型。
- **Megatron-LM**：
  - 在模型并行方面表现出色，特别是在处理超大规模模型时。
  - 对 NVIDIA GPU 进行了深度优化，性能更佳。

### 6. **合作与集成**
- **DeepSpeed 和 Megatron-LM**：
  - 两者可以结合使用，DeepSpeed 的数据并行与 Megatron-LM 的模型并行相结合，能够实现更高效的训练。
  - 例如，在训练 BLOOM 模型时，结合了 Megatron-LM 的张量并行和 DeepSpeed 的 Zero 优化。

### 总结
- **DeepSpeed** 更适合需要高性能数据并行和丰富优化功能的场景，尤其是在内存优化和训练速度方面表现出色。
- **Megatron-LM** 更适合需要高效模型并行和深度 GPU 优化的场景，尤其是在处理超大规模模型时表现出色。


## interleaving pipelin
## ring attention 和 cp dp
## pipe dream ，pipe dream flash，EFEB
## interleaved pipeline  efeb
## efeb，降低峰值显存
## interleaved pipeline减少水泡
## MAR 
## NCCL 
## DP FSDP zero 
## GPU

| GPU型号 | 速度单位 |
|:--------|:---------|
| A100    | 1        |
| H100    | 2-3      |
| A800    | 约0.8    |
| H800    | 约1.5    |

1. **H100**：H100采用更先进的Hopper架构，显存带宽高达3.35TB/s，相比A100的2TB/s有显著提升。在混合精度训练中，H100的性能大约是A100的2-3倍。
2. **A800**：A800是基于A100的“阉割版”，主要限制了NVLink互联带宽。其训练速度略低于A100，大约为A100的0.8倍。
3. **H800**：H800是基于H100的“阉割版”，限制了部分带宽。尽管受限，其性能仍然高于A100，大约为A100的1.5倍。

