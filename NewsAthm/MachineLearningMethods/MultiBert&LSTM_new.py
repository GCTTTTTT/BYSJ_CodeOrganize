# 注意这个是Bi-LSTM
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# 加载数据
# file_path = '../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled_Example.csv'
file_path = '../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv'

data = pd.read_csv(file_path,low_memory=False,lineterminator="\n")

# 加载BERT tokenizer和模型
model_name = '../bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# 将模型移动到GPU
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

# 定义训练和评估函数
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device, label_map):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.tolist())
            true_labels.extend(labels.tolist())

    label_map_inv = {v: k for k, v in label_map.items()}
    predictions = [label_map_inv[i] for i in predictions]
    true_labels = [label_map_inv[i] for i in true_labels]

    report = classification_report(true_labels, predictions, digits=4)
    return report


# 设置超参数
num_epochs = 20
batch_size = 16
learning_rate = 2e-5
hidden_size = 128
num_classes = len(label_map)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# 存储所有fold的性能指标
all_reports = []

# K-Fold交叉验证
for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    print(f'Fold {fold + 1}')
    
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    train_texts = train_data['body'].tolist()
    train_labels = train_data['category1'].map(label_map).tolist()
    val_texts = val_data['body'].tolist()
    val_labels = val_data['category1'].map(label_map).tolist()

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LSTMClassifier(bert_model, hidden_size, num_classes)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_report = evaluate(model, val_loader, device, label_map)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print('Validation Report:')
        print(val_report)

        val_loss = 1 - float(val_report.split('\n')[-2].split()[-2])  # 提取验证集损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')

    # 在每个fold结束后,评估最佳模型在验证集上的性能
    best_model = LSTMClassifier(bert_model, hidden_size, num_classes)
    best_model.load_state_dict(torch.load(f'best_model_fold_{fold + 1}.pth'))
    best_model.to(device)
    val_report = evaluate(best_model, val_loader, device, label_map)
    all_reports.append(val_report)

    print(f'Fold {fold + 1} Best Validation Report:')
    print(val_report)
    print()
    
