import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report


# 加载数据
file_path = '../../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv'

data = pd.read_csv(file_path,low_memory=False,lineterminator="\n")


nan_check = data['body'].isna().sum()
nan_check_c = data['category1'].isna().sum()
print(nan_check)
print(nan_check_c)

data = data.dropna(subset=['category1','body'])
nan_check = data['body'].isna().sum()
nan_check_c = data['category1'].isna().sum()
print(nan_check)
print(nan_check_c)


# 将孟加拉语类别转换为数字标签
label_map = {label: i for i, label in enumerate(data['category1'].unique())}
data['label'] = data['category1'].map(label_map)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['body'], data['label'], test_size=0.2, random_state=42)

# 使用TF-IDF对文本进行向量化
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# KNN模型
knn = KNeighborsClassifier(n_neighbors=33)
knn.fit(X_train_tfidf, y_train)
y_pred_knn = knn.predict(X_test_tfidf)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))



# SVM模型
svm = SVC()
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# RandomForest模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)
print("RandomForest Classification Report:")
print(classification_report(y_test, y_pred_rf))




# XGBOOST模型
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train_tfidf, y_train)
y_pred_xgb = xgb.predict(X_test_tfidf)
print("XGBOOST Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# LightGBM模型
lgbm = LGBMClassifier(n_estimators=100, random_state=42)
lgbm.fit(X_train_tfidf, y_train)
y_pred_lgbm = lgbm.predict(X_test_tfidf)
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgbm))