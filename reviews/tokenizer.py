from transformers import AutoTokenizer



# 加载预训练模型的分词器
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")


# 定义特殊标记
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# 示例文本
# text = "Hello, how are you?"
text = "我喜欢你"


# 编码文本
inputs = tokenizer.encode(
    text,
    add_special_tokens=True,  # 添加特殊标记
    return_tensors="pt"       # 返回 PyTorch 张量
)

# 输出编码后的输入
print(inputs)
for input_id in inputs[0]:
    print(tokenizer.decode(input_id.item()))