# BERT 中文项目
本项目基于 `bert-base-chinese` 实现中文文本处理功能。

## 模型说明
因模型权重文件（392.51MB）超出GitHub单文件100MB限制，未上传至仓库。
### 自动下载
运行代码时，通过`transformers`库自动下载：
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
```
