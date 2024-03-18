import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# Function to calculate quadratic loss
def quadratic_loss(predictions, actuals):
    loss = 0
    for i in range(1, 5):
        predicted = predictions[:, i - 1] #取出第i行資料
        actual = (actuals == i).astype(int) #相同為1反之為0
        loss += np.sum((predicted - actual) ** 2)
    return loss / len(actuals)

# Function to calculate informational loss
def informational_loss(predictions, actuals):
    loss = 0
    for i in range(1, 5):
        predicted = predictions[:, i - 1]
        actual = (actuals == i).astype(int)
        epsilon = 1e-15
        loss += -np.sum(actual * np.log(predicted + epsilon))
    return loss / len(actuals)

# Function to create a confusion matrix
def create_confusion_matrix(predictions, actuals):
    a=[0,0,0,0]
    predicted_classes = np.argmax(predictions, axis=1) + 1
    # print(predicted_classes)
    # for k in range(0,300):
    #     if predicted_classes[k] == 1:
    #       a[0] = a[0]+1
    #     elif predicted_classes[k] == 2:
    #       a[1] = a[1]+1
    #     elif predicted_classes[k] == 3:
    #       a[2] = a[2]+1
    #     elif predicted_classes[k] == 4:
    #       a[3] = a[3]+1
    # print(a)
    return confusion_matrix(actuals, predicted_classes, labels=[1, 2, 3, 4])

# Function to calculate F1 score for multi-class classification
def calculate_f1_score(predictions, actuals):
    predicted_classes = np.argmax(predictions, axis=1) + 1
    return f1_score(actuals, predicted_classes, average='macro')

# Load the data
data = pd.read_excel('homework_2023nov20.xlsx')

data = data.drop(0)

data = data.rename(columns={
    'Unnamed: 0': 'id',
    'Unnamed: 1': 'actual',
    '甲分類器': '甲_1',
    'Unnamed: 3': '甲_2',
    'Unnamed: 4': '甲_3',
    'Unnamed: 5': '甲_4',
    '乙分類器': '乙_1',
    'Unnamed: 8': '乙_2',
    'Unnamed: 9': '乙_3',
    'Unnamed: 10': '乙_4'
})


data['actual'] = data['actual'].astype(int)
print(data)


# Extracting predictions for each classifier
甲_predictions = data[['甲_1', '甲_2', '甲_3', '甲_4']].to_numpy()
乙_predictions = data[['乙_1', '乙_2', '乙_3', '乙_4']].to_numpy()
# print(甲_predictions)
# print(乙_predictions)

# Calculating quadratic loss for each classifier
quadratic_loss_甲 = quadratic_loss(甲_predictions, data['actual'])
quadratic_loss_乙 = quadratic_loss(乙_predictions, data['actual'])

# print("# Calculating quadratic loss for each classifier")
# print(quadratic_loss_甲)
# print(quadratic_loss_乙)


# Calculating informational loss for each classifier
informational_loss_甲 = informational_loss(甲_predictions, data['actual'])
informational_loss_乙 = informational_loss(乙_predictions, data['actual'])
# print("# Calculating informational loss for each classifier")
# print(informational_loss_甲)
# print(informational_loss_乙)

# Creating confusion matrices for both classifiers
confusion_matrix_甲 = create_confusion_matrix(甲_predictions, data['actual'])
confusion_matrix_乙 = create_confusion_matrix(乙_predictions, data['actual'])
# print(confusion_matrix_甲)
# print(confusion_matrix_乙)

# Calculating F1 scores for both classifiers
f1_score_甲 = calculate_f1_score(甲_predictions, data['actual'])
f1_score_乙 = calculate_f1_score(乙_predictions, data['actual'])

# Output results
print("第一題答案")
print("甲分類器的 Quadratic Loss:", quadratic_loss_甲)
print("乙分類器的 Quadratic Loss:", quadratic_loss_乙,"\n\n")
print("第二題答案")
print("甲分類器的 Informational Loss:", informational_loss_甲)
print("乙分類器的 Informational Loss:", informational_loss_乙,"\n\n")

print("第五題答案")
print("甲分類器的 Confusion Matrix:\n", confusion_matrix_甲,"\n\n")
print("第六題答案")
print("乙分類器的 Confusion Matrix:\n", confusion_matrix_乙,"\n\n")
print("第七題答案")
print("甲分類器的 F1 Score:", f1_score_甲)
print("乙分類器的 F1 Score:", f1_score_乙)
