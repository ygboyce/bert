
from transformers import (BertPreTrainedModel, BertConfig,
                          BertForSequenceClassification, BertTokenizer,BertModel,
                          )

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



data_path = "jiudian.txt"
bert_path = 'bert-base-chinese'

##########################################
bert = BertModel.from_pretrained(bert_path)
print(get_parameter_number(bert))
# print(get_parameter_number(model.bert.embeddings))
# print(get_parameter_number(model.bert.encoder.layer[0].attention))

dim = 768
emb_para = 768*2 + 768*512 + 768*21128
self_att = 768*768*3 + 768*768
mlp = 768*3072 + 3072*768
bertlayers_para = 12 * (self_att + mlp)

# print(768*768*3 + 768*768)
print(bertlayers_para+emb_para)

# for name, param in bert.named_parameters():
#     print(name, param.shape)

