import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 讀取訓練數據和測試數據
training_data_path = 'training_data.csv'  # 設定訓練數據文件的路徑
testing_data_path = 'testing_data.csv'    # 設定測試數據文件的路徑
training_data = pd.read_csv(training_data_path)
testing_data = pd.read_csv(testing_data_path)

# 分離特徵和標籤
X = training_data.drop(['id', 'label'], axis=1)
y = training_data['label']

# 將訓練數據分為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化並訓練決策樹分類器
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 在驗證集上評估分類器
y_pred = dt_classifier.predict(X_val)
print(classification_report(y_val, y_pred))

# 使用分類器對測試數據進行預測
X_test = testing_data.drop('id', axis=1)
test_predictions = dt_classifier.predict(X_test)

# 將預測結果保存到CSV文件
submission = pd.DataFrame({'id': testing_data['id'], 'label': test_predictions})
submission_file_path = 'submission_predictions.csv'
submission.to_csv(submission_file_path, index=False)
