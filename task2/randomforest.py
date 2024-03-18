import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTETomek
# Load the training and testing data
training_data = pd.read_csv('balanced_training_data.csv')
testing_data = pd.read_csv('testing_data.csv')


# Fill missing values with the mean of each column
training_data_filled = training_data.fillna(training_data.mean())
testing_data_filled = testing_data

# Separate the features and the target variable
X = training_data_filled.drop('label', axis=1)
y = training_data_filled['label']

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smt = SMOTETomek(random_state=42)
# X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)

# Initialize and train the Random Forest Classifier
# rf_classifier = RandomForestClassifier(random_state=42)
# rf_classifier.fit(X_train, y_train)

# # Predict on the test set
# y_pred = rf_classifier.predict(X_test)

# # Evaluate the classifier
# accuracy = accuracy_score(y_test, y_pred)
# classification_report_result = classification_report(y_test, y_pred)

# # Print the accuracy and classification report
# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(classification_report_result)

X_train_balanced, y_train_balanced = smt.fit_resample(X, y)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X, y)
# Predict on the test set
y_pred = rf_classifier.predict(testing_data)
submission_df = pd.DataFrame({'id': testing_data['id'], 'label': y_pred})
submission_df.to_csv('submission.csv', index=False)
