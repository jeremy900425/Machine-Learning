import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

# 加載訓練和測試數據
training_data = pd.read_csv('training_data.csv')
testing_data = pd.read_csv('testing_data.csv')

# 用每列的平均值填充訓練數據中的缺失值
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(training_data.drop(['label','id'], axis=1))
y_train = training_data['label']

# 對特徵進行標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 使用 SMOTE + Tomek Links 平衡訓練數據
smt = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smt.fit_resample(X_train_scaled, y_train)

# 初始化 SVM 分類器
svm_classifier = SVC()

# 使用平衡的訓練數據訓練分類器
svm_classifier.fit(X_train_balanced, y_train_balanced)

# 對測試數據進行同樣的預處理
X_test = imputer.transform(testing_data.drop('id', axis=1))
X_test_scaled = scaler.transform(X_test)

# 使用 SVM 分類器進行預測
test_predictions = svm_classifier.predict(X_test_scaled)

# 準備提交文件
submission_df = pd.DataFrame({'id': testing_data['id'], 'label': test_predictions})
submission_df.to_csv('submission.csv', index=False)
