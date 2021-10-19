import warnings
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate


# Read the csv file

readdata = pd.read_csv('drug200.csv')

# Plot the data of csv file


drug_group = [x for x in readdata['Drug']]
drug_type_count = []
drug_type = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
for x in drug_type:
    drug_type_count.append(drug_group.count(x))
x = np.array(drug_type)
y = np.array(drug_type_count)
plt.bar(x, y, color="#6b9fff", width=0.4, edgecolor='#ff0000')
plt.savefig('Drug-distributiontask2abc.pdf', dpi=370)

# Convert all ordinal values to numeric values

replace_Drug = {'drugA': 1, 'drugB': 2, 'drugC': 3, 'drugX': 4, 'drugY': 5}
readdata.replace(replace_Drug, inplace = True)
#print(readdata)

replace_Sex={'F': 1, 'M': 0}
readdata.replace(replace_Sex, inplace = True)
#print(readdata)

replace_BP = {'LOW': 1, 'NORMAL': 2, 'HIGH': 3}
readdata.replace(replace_BP, inplace = True)
#print(readdata)

replace_Cholestrol = {'LOW': 1, 'NORMAL': 2, 'HIGH': 3}
readdata.replace(replace_Cholestrol, inplace = True)
#print(readdata)

#

a, b = readdata.iloc[:, :-1], readdata.iloc[:, -1]

#print(a)

#print(b)

# split the data set
a_training, a_testing, b_training, b_testing = train_test_split(a, b)


# Task2_a,b,c Run 3 different classifiers
openfile = open("drug-performancetask2abc.txt", 'w')
X = a_training
Y = b_training
test_X = a_testing

class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
feature_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

# (a) NB: a Gaussian Naive Bayes Classifier (naive bayes.GaussianNB) with the default parameters.
openfile.write("(a) ---------------- GaussianNB default values-------------------\n")
clf_gaussiannb = GaussianNB()
clf_gaussiannb.fit(X, Y)
predict_gaussiannb = clf_gaussiannb.predict(test_X)

matrix_nb = confusion_matrix(b_testing, predict_gaussiannb)
openfile.write("-----------------Confusion Matrix-----------------\n")
confusionMatrix = pd.DataFrame(matrix_nb, index=class_columns)
openfile.write(tabulate(confusionMatrix, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

classification_gaussiannb = classification_report(b_testing, predict_gaussiannb, target_names=class_columns)

openfile.write(classification_gaussiannb)
openfile.write("\n")

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy_score(b_testing, predict_gaussiannb))
macro_f1 = str(f1_score(b_testing, predict_gaussiannb, average='macro'))
weighted_f1 = str(f1_score(b_testing, predict_gaussiannb, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write(tabulate(displayed_data, tablefmt="pipe"))
openfile.write("\n")

# (b) Base-DT: a Decision Tree (tree.DecisionTreeClassifier) with the default parameters
openfile.write("\n(b) ---------------- Base-DT default values-------------------\n")
clf_basedt = DecisionTreeClassifier()
clf_basedt.fit(X, Y)
predict_basedt = clf_basedt.predict(test_X)

matrix_nb = confusion_matrix(b_testing, predict_basedt)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
openfile.write("The Confusion Matrix\n")
confusionMatrix = pd.DataFrame(matrix_nb, index=class_columns)
openfile.write(tabulate(confusionMatrix, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

classification_basedt = classification_report(b_testing, predict_basedt, target_names=class_columns)

openfile.write(classification_basedt)
openfile.write("\n")

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy_score(b_testing, predict_basedt))
macro_f1 = str(f1_score(b_testing, predict_basedt, average='macro'))
weighted_f1 = str(f1_score(b_testing, predict_basedt, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write(tabulate(displayed_data, tablefmt="pipe"))
openfile.write("\n")

#(c) -----------------Top-DT--------------------
openfile.write("\n(c) -----------------Top-DT--------------------\n")
# Experiment 1  Entropy + max_depth(3) + min_sample_split(2)
top_DT_parameters = {'criterion': ('entropy', 'gini'),
                'max_depth': (5, 10), 'min_samples_split': (2, 4, 6)}
clf_topdt = GridSearchCV(DecisionTreeClassifier(), top_DT_parameters)
clf_topdt.fit(X, Y)
# predict with the best found params
predict_topdt = clf_topdt.predict(test_X)


matrix_nb = confusion_matrix(b_testing, predict_topdt)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
openfile.write("The Confusion Matrix\n")
confusionMatrix = pd.DataFrame(matrix_nb, index=class_columns)
openfile.write(tabulate(confusionMatrix, class_columns, tablefmt="pipe", stralign='center'))
openfile.write('\n')

classification_topdt = classification_report(b_testing, predict_topdt, target_names=class_columns)
openfile.write(classification_topdt)
openfile.write("\n")

rows = ["accuracy", "macro-average F1", "weighted-average F1"]
accuracy = str(accuracy_score(b_testing, predict_topdt))
macro_f1 = str(f1_score(b_testing, predict_topdt, average='macro'))
weighted_f1 = str(f1_score(b_testing, predict_topdt, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], rows)
openfile.write(tabulate(displayed_data, tablefmt="pipe"))


accuracy = []
macro_f1 = []
weighted_f1 = []
for i in range(10):

    # creating new models
    top_DT_param = {'criterion': ('entropy', 'gini'),
                    'max_depth': (2, 7), 'min_samples_split': (3, 6, 8)}

    clf = [GaussianNB(),
           DecisionTreeClassifier(),
           GridSearchCV(DecisionTreeClassifier(), top_DT_param),
           ]
    p_result = []
    accuracy_temp = []
    macro_temp = []
    weighted_temp = []

    # training and testing new models
    for j in range(3):
        # training
        clf[j].fit(X, Y)
        # testing
        p_result.append(clf[j].predict(test_X))

    # recording scores
    for k in range(3):
        # accuracy
        accuracy_temp.append(accuracy_score(b_testing, p_result[k]))
        accuracy.append(accuracy_temp)
        # macro F1
        macro_temp.append(f1_score(b_testing, p_result[k], average='macro'))
        macro_f1.append(macro_temp)
        # weighted F1
        weighted_temp.append(f1_score(b_testing, p_result[k], average='weighted'))
        weighted_f1.append(weighted_temp)

# calculate the average and stddev
accuracy = np.array(accuracy)
macro_f1 = np.array(macro_f1)
weighted_f1 = np.array(weighted_f1)

mean_accuracy = np.mean(accuracy, axis=0)
mean_macro_f1 = np.mean(macro_f1, axis=0)
mean_weighted_f1 = np.mean(weighted_f1, axis=0)
std_accuracy = np.std(accuracy, axis=0)
std_macro_f1 = np.std(macro_f1, axis=0)
std_weighted_f1 = np.std(weighted_f1, axis=0)

columns = ["GaussianNB", "BaseDT", "TopDT"]
rows = ["mean_accuracy", "mean_macro_f1", "mean_weighted_f1", "std_accuracy", "std_macro_f1", "std_weighted_f1"]
t10average = pd.DataFrame([mean_accuracy,
                     mean_macro_f1,
                     mean_weighted_f1,
                     std_accuracy,
                     std_macro_f1,
                     std_weighted_f1],
                    rows)
openfile.write('\n')
openfile.write('\nTen times average\n')
openfile.write('\n')
openfile.write(tabulate(t10average, columns, tablefmt="pipe"))

# Close the file
openfile.close()

