import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm


# 加载数据
# file_path = '../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled_Example.csv'
file_path = '../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv'

data = pd.read_csv(file_path,low_memory=False,lineterminator="\n")

# 加载BERT tokenizer和模型
model_name = '../bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)


# 将模型移动到GPU(如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# 定义数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 将孟加拉语类别转换为数字标签
label_map = {label: i for i, label in enumerate(data['category1'].unique())}
labels = data['category1'].map(label_map).tolist()

# 将数据划分为训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(data['body'].tolist(), labels, test_size=0.2, random_state=42)
print("1")
# 创建数据集和数据加载器
train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
print("1")

# 定义LSTM分类器
class LSTMClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes, num_layers=1, bidirectional=True, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.bert = bert_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(bert_model.config.hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lstm_output, _ = self.lstm(last_hidden_state)
        lstm_output = self.dropout(lstm_output[:, -1, :])
        logits = self.fc(lstm_output)
        return logits

# 创建LSTM分类器实例
hidden_size = 128
num_classes = len(label_map)
lstm_classifier = LSTMClassifier(bert_model, hidden_size, num_classes)
lstm_classifier.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_classifier.parameters(), lr=2e-5)
print("1")

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    lstm_classifier.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = lstm_classifier(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

# 在测试集上评估模型
lstm_classifier.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = lstm_classifier(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# 将数字标签转换回孟加拉语类别
label_map_inv = {i: label for label, i in label_map.items()}
test_preds = [label_map_inv[i] for i in test_preds]
test_labels = [label_map_inv[i] for i in test_labels]

# 评估模型性能
print("LSTM Classifier:")
print(classification_report(test_labels, test_preds))
