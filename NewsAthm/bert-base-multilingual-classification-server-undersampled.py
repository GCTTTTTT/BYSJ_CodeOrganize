# undersampled 欠采样平衡后数据进行训练 ephoch=10  使用bert-base-multilingual-cased
# update：3.8 FIX2
# update：3.13 尝试train：test 0.85:0.15
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import logging

def reset_log(log_path):
    import logging
    fileh = logging.FileHandler(log_path, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.INFO)

reset_log('./logs_FIX2/085_bert-base-multilingual-cased-classification-server-undersampled_training.log')
logger = logging.getLogger(__name__)
logging.info('This is a log info')

# # 配置日志记录器
# logging.basicConfig(filename='./logs/bert-classification-server-undersampled_training.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') ## 



# 读取CSV文件
# file_path = './datasets/mangoNews_Example.csv'  # 测试版数据集（规模小5m）
# file_path = './datasets/mangoNews.csv'          # 完整版数据集（13g） 需要分块读
# file_path = './datasets/mangoNews_Example_100000.csv'          # 100000行数据集（2.2g）
# file_path = './datasets/mangoNews_Example_10000.csv'          # 10000行数据集（190m)
# file_path = './datasets/deduplicated_mangoNews_Nums3000p_CategoryMerge.csv'          # 去重后保留类别对应语料数在3000+并合并同义类别的数据集（9.9g) 需要分块读
# file_path = './datasets/deduplicated_mangoNews_Nums3000p_CategoryMerge_100000_1.csv'          # 去重后保留类别对应语料数在3000+并合并同义类别的数据集（576m) 
# file_path = './datasets/deduplicated_mangoNews_Nums3000p_CategoryMerge_990000.csv'          # 去重后保留类别对应语料数在3000+并合并同义类别的数据集（8.1g) 
# file_path = './datasets/deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv'          # 去重后保留类别对应语料数在3000+并合并同义类别(新）并随机欠采样数据平衡的数据集（12095*12条 1.2g) 

# file_path = './datasets_FIX/FIX_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv' # new
file_path = './datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv' # new


data = pd.read_csv(file_path,low_memory=False,lineterminator="\n")

# Select relevant columns
data = data[['body', 'category1']]

# 统计 'category1' 列中每种类别的个数
category_counts = data['category1'].value_counts()

# 设置显示选项，完整输出结果
pd.set_option('display.max_rows', None)
print("Category Counts:")
print(category_counts)
# 恢复默认显示选项
pd.reset_option('display.max_rows')

# 将类别列转换为整数标签  注意是data['category1']
label_to_id = {label: idx for idx, label in enumerate(data['category1'].unique())}
print(label_to_id)
data['label'] = data['category1'].map(label_to_id)
num_classes = len(label_to_id)
print(num_classes)

# 划分训练集和测试集
# train_data, test_data = train_test_split(data, test_size=0.3, random_state=42) ## 2.20 test_size:0.2->0.3
train_data, test_data = train_test_split(data, test_size=0.15, random_state=42) ## 3.13 test_size:0.3->0.15


# 检验 'category1' 列是否还有 NaN 值
# nan_check = train_data['category1'].isnull().sum()
nan_check = train_data['category1'].isna().sum()

print("before")
if nan_check > 0:
    print(f"There are still {nan_check} NaN values in the 'category1' column.")
else:
    print("No NaN values in the 'category1' column.")
    
# 去除包含缺失值的样本
train_data = train_data.dropna(subset=['category1'])
test_data = test_data.dropna(subset=['category1'])
# todo：注意后面要提取缺失值

print("after")
# 检验 'category1' 列是否还有 NaN 值
# nan_check = train_data['category1'].isnull().sum()
nan_check = train_data['category1'].isna().sum()

if nan_check > 0:
    print(f"There are still {nan_check} NaN values in the 'category1' column.")
else:
    print("No NaN values in the 'category1' column.")

# Bert_path = './uncased_L-12_H-768_A-12'  # bert-base-uncased from github
# Bert_path = './wwm_uncased_L-24_H-1024_A-16'  # bert-large-uncased(whole word masking) from github
Bert_path =  './bert-base-multilingual-cased' # bert-base-multilingual-cased

# 初始化Bert tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(Bert_path) # bert: base or large（wwm&origin） 

print(tokenizer.tokenize('I have a good time, thank you.')) # 测试

num_classes = len(data['category1'].unique())  # num_labels 表示分类任务中唯一类别的数量
model = BertForSequenceClassification.from_pretrained(Bert_path, num_labels=num_classes)

model_name = "bert-base-multilingual-cased" ## 
# num_classes
# ======================================================================================================================================================
# # 继续版需要修改
# # 加载已有模型！！
# # 加载之前保存的模型参数
# # 2.22 0:25 e10
# model.load_state_dict(torch.load('./models/Bert-multilingual/bert-base-multilingual-cased_classification_undersampled_new_epoch_10.pth')) ## !!!!
# # ======================================================================================================================================================


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        # label = str(self.labels[idx]) # todo: 改str试试

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long) 
        }
    
    
# 创建训练和测试数据集实例
train_dataset = CustomDataset(train_data['body'].values, train_data['label'].values, tokenizer)
test_dataset = CustomDataset(test_data['body'].values, test_data['label'].values, tokenizer)

# 使用DataLoader加载数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 2.20 32-64 ## 64
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 2.20 32-64

# 将模型移动到GPU上（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


# 定义优化器和损失函数
# optimizer = AdamW(model.parameters(), lr=2e-5) # 优化器可调整 学习率可调整
optimizer = torch.optim.AdamW(model.parameters(),  lr=2e-5) # 修改新用法  优化器可调整 学习率可调整
criterion = torch.nn.CrossEntropyLoss() # 损失函数可调整

# 定义训练函数
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    with tqdm(loader, desc="Training", leave=False) as iterator:
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            iterator.set_postfix(loss=loss.item())

    return total_loss / len(loader)


# 定义评估函数  ## 
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    #2.20改
    # 修改返回val_loss
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad(), tqdm(loader, desc="Evaluating", leave=False) as iterator:
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            # 修改返回val_loss
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            num_samples += len(labels)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 修改返回val_loss
    avg_loss = total_loss / num_samples
    return all_labels, all_preds, avg_loss


# 训练模型
# ======================================================================================================================================================
# continue版要改
num_epochs = 20
# ======================================================================================================================================================

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    #2.20改
    t1,t2,val_loss = evaluate(model, test_loader, device)  # 假设evaluate函数用于计算验证集损失 ## 
        # print(classification_report(true_labels, predicted_labels))
    print(classification_report(t1,t2,zero_division=1))
    logging.info(classification_report(t1,t2,zero_division=1)) ## 

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}') ## 
    # 在每个 epoch 结束时记录训练损失和验证损失
    logging.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')  ## 



    

# 评估模型
true_labels, predicted_labels,test_loss = evaluate(model, test_loader, device) ## 
print(true_labels[:100])
print(predicted_labels[:100])
# print(true_labels)
# print(predicted_labels)

# print(classification_report(true_labels, predicted_labels))
print(classification_report(true_labels, predicted_labels,zero_division=1))
logging.info(classification_report(true_labels, predicted_labels,zero_division=1)) ## 


# ======================================================================================================================================================
# # continue版要改
# # 2.22 0.26 already_epoch = 10 
# already_epoch = 10 ## 
# num_epochs += already_epoch

# ======================================================================================================================================================



# model_path = f"./models_FIX/{model_name}_classification_undersampled_new_epoch_{num_epochs}.pth"  ## 
model_path = f"./models_FIX2/085_{model_name}_classification_undersampled_new_epoch_{num_epochs}.pth"  ## 


# 保存模型到文件
torch.save(model.state_dict(), model_path)

# # 保存模型到文件
# torch.save(model.state_dict(), 'bert_model_undersampled_new_e10.pth')
