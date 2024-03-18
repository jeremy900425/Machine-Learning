import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

# Load the training and testing data
training_data = pd.read_csv('training_data.csv')
testing_data = pd.read_csv('testing_data.csv')

# training_data_filled = training_data.fillna(training_data.mean())
imputer = IterativeImputer(max_iter=10, random_state=0)
training_data_filled = imputer.fit_transform(training_data)
# 將插補後的數據轉換回 DataFrame
training_data_filled = pd.DataFrame(training_data_filled, columns=training_data.columns)
# print(training_data_filled)
# exit()
# Separate features and target variable from the training data
X_train = training_data_filled.drop('label', axis=1)
y_train = training_data_filled['label']
# print(X_train)
# print(y_train)
# exit()


# Apply SMOTE + Tomek Links to balance the training data
smt = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)
# X_train_balanced = X_train_balanced.round(3)
# print(X_train_balanced)
# print(y_train_balanced)
# print(testing_data)
# exit()


# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(use_label_encoder=True, eval_metric='logloss')

# Train the classifier using balanced training data
xgb_classifier.fit(X_train_balanced, y_train_balanced)

# test = testing_data.drop('id', axis=1)
# print(test)
# exit()

test_predictions = xgb_classifier.predict(testing_data)

submission_df = pd.DataFrame({'id': testing_data['id'], 'label': test_predictions})
submission_df.to_csv('submission.csv', index=False)
