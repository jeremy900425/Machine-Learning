from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
def preprocess_data(df):
    columns_to_remove = ['x0', 'x3', 'x9']
    
    df = df.drop(columns=columns_to_remove, errors='ignore')

    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col] = df[col].fillna(df[col].mean())
    
    # For categorical columns (like 'x1'), fill missing values with the mode (most frequent value)
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df
def balance_data_with_smotetomek(data):
    # 將非數值型特徵轉換為數值型
    label_encoder = LabelEncoder()
    data['x1'] = label_encoder.fit_transform(data['x1'])
    data['label'] = label_encoder.fit_transform(data['label'])

    # 分割特徵和標籤
    X = data.drop('label', axis=1)
    y = data['label']

    # 應用SMOTE-Tomek
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    balanced_label_distribution = pd.Series(y_resampled).value_counts()
    print(balanced_label_distribution)

    return X_resampled,y_resampled

def xgboost(x_train,y_train,data_test,sample_submission):
  
  from xgboost import XGBClassifier

  # 初始化XGBoost分類器
  xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(y_train.unique()))

  # 訓練XGBoost模型
  xgb_model.fit(x_train, y_train)

  # 預處理測試數據
  processed_test = preprocess_data(data_test)
  # 將非數值型特徵轉換為數值型
  label_encoder = LabelEncoder()
  processed_test['x1'] = label_encoder.fit_transform(processed_test['x1'])

  # 進行預測
  predictions = xgb_model.predict(processed_test)
  # 將預測結果從數字編碼轉換回原始標籤
  label_encoder = LabelEncoder()
  label_encoder.fit(data_train['label'])
  predicted_labels = label_encoder.inverse_transform(predictions)

  # 將預測結果與範例提交檔案結合
  sample_submission['label'] = predicted_labels

  # 保存預測結果
  sample_submission.to_csv('您的預測結果.csv', index=False)


def svm_classifier(x_train, y_train, data_test, sample_submission):
    # 初始化SVM分類器
    svm_model = SVC()

    # 訓練SVM模型
    svm_model.fit(x_train, y_train)

    # 預處理測試數據
    processed_test = preprocess_data(data_test)
    label_encoder = LabelEncoder()
    processed_test['x1'] = label_encoder.fit_transform(processed_test['x1'])

    # 進行預測
    predictions = svm_model.predict(processed_test)

    # 將預測結果從數字編碼轉換回原始標籤
    label_encoder = LabelEncoder()
    label_encoder.fit(data_train['label'])
    predicted_labels = label_encoder.inverse_transform(predictions)

    # 將預測結果與範例提交檔案結合
    sample_submission['label'] = predicted_labels

    # 保存預測結果
    sample_submission.to_csv('您的預測結果.csv', index=False)

# Usage example
# processed_data = preprocess_data(your_dataframe)
# 使用這個函數來處理數據
data_train = pd.read_csv('training_data.csv')
data_test = pd.read_csv('testing_data.csv')
sample_submission = pd.read_csv('sample_submission.csv')

processed_train = preprocess_data(data_train)

print(processed_train)

x_train,y_train=balance_data_with_smotetomek(processed_train)
print(x_train)
print(y_train)
# exit()
xgboost(x_train,y_train,data_test,sample_submission)

