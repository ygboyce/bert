from transformers import BertModel, BertTokenizer







bert = BertModel.from_pretrained("bert-base-chinese")
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(bert))


emb_num = 21128*768 + 2*768 + 512*768    #忽略了bias

self_att_num = 768*768*3+ 768*768 + 768*3072+3072*768
all_att_num = 12 * self_att_num

pool_num = 768*768
print(emb_num+ all_att_num + pool_num)
#
# for name, para in bert.named_parameters():
#     print(name, para.shape)

