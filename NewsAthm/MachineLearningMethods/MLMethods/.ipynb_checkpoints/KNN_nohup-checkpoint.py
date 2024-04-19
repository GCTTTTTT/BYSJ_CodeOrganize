import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer, BertModel

# 加载数据
# file_path = '../../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled_Example.csv'
file_path = '../../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv'

data = pd.read_csv(file_path,low_memory=False,lineterminator="\n")


# 加载微调好的BERT模型和tokenizer
model_name = '../../bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

nan_check = data['body'].isna().sum()
nan_check_c = data['category1'].isna().sum()
print(nan_check)
print(nan_check_c)

data = data.dropna(subset=['category1','body'])
nan_check = data['body'].isna().sum()
nan_check_c = data['category1'].isna().sum()
print(nan_check)
print(nan_check_c)


# 将模型移动到GPU(如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# # 使用BERT模型对新闻正文进行向量化
# def vectorize_text(text):
#     inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = bert_model(**inputs)
#     embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#     return embeddings
# 使用BERT模型对新闻正文进行向量化
def vectorize_text(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings



# # 对新闻正文进行向量化
# X = data['body'].apply(vectorize_text).tolist()
# 对新闻正文进行向量化
X = np.array([embeddings[0] for embeddings in data['body'].apply(vectorize_text)])
print("1")
label_map = {label: i for i, label in enumerate(data['category1'].unique())}
y = data['category1'].map(label_map).tolist()
print("2")
# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 创建K近邻分类器
knn_classifier = KNeighborsClassifier(n_neighbors=33)

# 训练K近邻分类器
knn_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = knn_classifier.predict(X_test)

# 评估模型性能
print(classification_report(y_test, y_pred))