import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

# Load the training and testing data
training_data = pd.read_csv('training_data.csv')
testing_data = pd.read_csv('testing_data.csv')

# Fill missing values with the mean of each column in the training data
training_data_filled = training_data.fillna(training_data.mean())
# columns_to_scale = training_data_filled.loc[:, 'col2':'col13']
# print(columns_to_scale)

# scaler = MinMaxScaler()

# 對選定的列進行正規化
# scaled_columns = scaler.fit_transform(columns_to_scale)
# print(scaled_columns)
# 將正規化後的數據放回原始 DataFrame
# training_data_filled.loc[:, 'col2':'col13'] = scaled_columns

# Separate features and target variable from the training data
X_train = training_data_filled.drop('label', axis=1)
y_train = training_data_filled['label']

# Apply SMOTE + Tomek Links to balance the training data
smt = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(use_label_encoder=True, eval_metric='logloss')

# Train the classifier using balanced training data
xgb_classifier.fit(X_train_balanced, y_train_balanced)

# columns_to_scale = testing_data.loc[:, 'col2':'col13']
# 對選定的列進行正規化
# scaled_columns = scaler.fit_transform(columns_to_scale)
# print(scaled_columns)
# 將正規化後的數據放回原始 DataFrame
# testing_data.loc[:, 'col2':'col13'] = scaled_columns

# Fill missing values in testing data and predict
test_predictions = xgb_classifier.predict(testing_data)

# Prepare submission file
submission_df = pd.DataFrame({'id': testing_data['id'], 'label': test_predictions})
submission_df.to_csv('submission.csv', index=False)
