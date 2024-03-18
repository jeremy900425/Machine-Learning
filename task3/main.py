import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 讀取數據
train_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('testing_data.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# 移除噪聲特徵
features_to_drop = ['x16', 'x10']
train_data = train_data.drop(features_to_drop, axis=1)
test_data = test_data.drop(features_to_drop, axis=1)

# 僅對數字型特徵填補缺失值
numeric_cols = train_data.select_dtypes(include=['number']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())

# 分割特徵和標籤
X = train_data.drop('label', axis=1)
y = train_data['label']

# 計算類別數量
num_classes = len(y.unique())

# 將標籤轉換為數字
y = pd.factorize(y)[0]

# 分割訓練和驗證數據集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練XGBoost模型
model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)
model.fit(X_train, y_train)

# 驗證模型
predictions = model.predict(X_val)
print(f"Accuracy: {accuracy_score(y_val, predictions)}")

# 預測測試數據集
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].mean())
test_predictions = model.predict(test_data)

# 創建提交文件
submission = pd.DataFrame({'id': sample_submission['id'], 'label': test_predictions})
submission['label'] = submission['label'].apply(lambda x: train_data['label'].unique()[x]) # 將數字轉換回標籤
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")

# https://chat.openai.com/share/2024c970-994e-4bbc-bf67-ff7bb51543db