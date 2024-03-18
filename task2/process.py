import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.impute import KNNImputer
# Load the training data

training_data = pd.read_csv('training_data.csv')
testing_data = pd.read_csv('testing_data.csv')

# 使用平均值填充缺失值
data_filled = training_data.fillna(training_data.mean())
# data_filled = training_data.fillna(0)
columns_to_scale = data_filled.loc[:, 'col2':'col13']
# print(columns_to_scale)

# scaler = StandardScaler()
scaler = MinMaxScaler()
# 對選定的列進行正規化
scaled_columns = scaler.fit_transform(columns_to_scale)
# print(scaled_columns)
# 將正規化後的數據放回原始 DataFrame
data_filled.loc[:, 'col2':'col13'] = scaled_columns

# print(data_filled)
# exit()
# 分離特徵和標籤
X = data_filled.drop('label', axis=1)
y = data_filled['label']

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)
# 查看訓練集三種類別比例
print(pd.Series(y_train).value_counts(normalize=True))
# 查看測試集三種類別比例
print(pd.Series(y_test).value_counts(normalize=True))

# 打印分割前的類別分佈
print("Original class distribution:", Counter(y_train))
print("Original class distribution:", Counter(y_test))

# 應用 SMOTE + Tomek Links
smt = SMOTETomek(sampling_strategy=1,random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)

# 打印應用 SMOTE + Tomek Links 後的類別分佈
print("Resampled class distribution:", Counter(y_res))

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 使用平衡後的數據訓練模型
model.fit(X_res, y_res)

# 使用驗證集進行預測
y_pred = model.predict(X_test)

# 計算準確度和 ROC AUC 分數
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# 計算混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)

# 計算 F1 分數
f1 = f1_score(y_test, y_pred)

# 打印結果
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)
print("F1 Score:", f1)




