#  data负责产生两个dataloader
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split   #给X,Y 和分割比例， 分割出来一个训练集和验证机的X, Y
import torch


def read_file(path):
    data = []
    label = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i > 200 and i< 7500:
                continue
            line = line.strip("\n")
            line = line.split(",", 1)  #把这句话，按照，分割， 1表示分割次数
            data.append(line[1])
            label.append(line[0])
    print("读了%d的数据"%len(data))
    return data, label


# file = "../jiudian.txt"
# read_file(file)
class jdDataset(Dataset):
    def __init__(self, data, label):
        self.X = data
        self.Y = torch.LongTensor([int(i) for i in label])

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.Y)





def get_data_loader(path, batchsize, val_size=0.2):          #读入数据，分割数据。
    data, label = read_file(path)
    train_x, val_x, train_y, val_y = train_test_split(data, label, test_size=val_size, shuffle=True, stratify=label)
    train_set = jdDataset(train_x, train_y)
    val_set = jdDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batchsize, shuffle=True)
    val_loader = DataLoader(val_set, batchsize, shuffle=True)
    return train_loader, val_loader

if __name__ == "__main__":
    get_data_loader("../jiudian.txt", 2)