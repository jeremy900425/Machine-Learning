import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 讀取訓練和測試數據
train_data_path = 'training_data.csv'  # 請替換為您的訓練數據文件路徑
test_data_path = 'testing_data.csv'    # 請替換為您的測試數據文件路徑
sample_submission_path = 'sample_submission.csv' # 請替換為您的範例提交文件路徑

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# 分離特徵和標籤
X_train = train_data.drop(['id', 'label'], axis=1)
y_train = train_data['label']

# 標準化數據
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# 訓練SVM模型
svm_model = SVC()
svm_model.fit(X_train, y_train)

# 對測試數據進行預測
X_test = test_data.drop('id', axis=1)
# X_test_scaled = scaler.transform(X_test)
test_predictions = svm_model.predict(X_test)

# 創建提交文件
submission = pd.DataFrame({'id': test_data['id'], 'label': test_predictions})
submission.to_csv('svm_submission.csv', index=False)

print("Submission file created successfully.")
