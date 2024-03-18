import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
# 讀取訓練數據和測試數據
training_data_path = 'training_data.csv'  
testing_data_path = 'testing_data.csv'    
training_data = pd.read_csv(training_data_path)
testing_data = pd.read_csv(testing_data_path)

# 分離特徵和標籤
X = training_data.drop(['id', 'label'], axis=1) #取得feature
y = training_data['label'] # 取得label
# print(X)
# print(y)
# exit()

# 將訓練數據分為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) #分（80 20）

# print(X_train)
# print(X_val)

# print(y_train)
# print(y_val)
# exit()

# 初始化並訓練隨機森林
rf_classifier = RandomForestClassifier(n_estimators=200, 
                                       max_depth=20, 
                                       min_samples_split=5, 
                                       min_samples_leaf=2,
                                       random_state=42)
rf_classifier.fit(X_train, y_train)

# 在驗證集上評估分類器
y_pred = rf_classifier.predict(X_val)
print(classification_report(y_val, y_pred))

# 使用分類器對測試數據進行預測
X_test = testing_data.drop('id', axis=1)
test_predictions = rf_classifier.predict(X_test)

# 指定文件的路徑
file_path = 'submission_predictions.csv'

# 檢查檔案是否存在
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File {file_path} has been deleted.")
else:
    print(f"File {file_path} does not exist.")


# 將預測結果保存到CSV文件
submission = pd.DataFrame({'id': testing_data['id'], 'label': test_predictions})
submission_file_path = 'submission_predictions.csv'
submission.to_csv(submission_file_path, index=False)
