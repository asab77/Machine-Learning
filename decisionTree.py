# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt

# load dataset

pima_df = pd.read_csv("diabetes.csv")


pima_df.head()

pima_df.describe()

pima_df.info()

shape = pima_df.shape
print(shape)

pima_df.isnull().sum()


# Features
X = pima_df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]

# Target variable
y = pima_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 1)

clf = DecisionTreeClassifier(criterion="entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)

# Train and Fit Decision Tree Classifer
clf = clf.fit(X_train , y_train)

#Predict the response for test dataset
dtree_y_pred = clf.predict(X_test)

# Report the Accuracy, Recall, Precision and F1 score for the predictions
dtree_y_true = y_test

accuracy = metrics.accuracy_score(dtree_y_true, dtree_y_pred)
precision = metrics.precision_score(dtree_y_true,dtree_y_pred, average = "weighted")
recall = metrics.recall_score(dtree_y_true, dtree_y_pred, average ="weighted")
f1_score = metrics.f1_score(dtree_y_true, dtree_y_pred, average = "weighted")#

print("Accuracy:", accuracy)
print("Precision Score:", precision)
print("Recall Score: ", recall) 
print("F1 Score: ", f1_score) 

# Here a template for writing the function for getting the tpr, fpr and threshold as well as AUC values is given as well as a simple way to plot the ROC curve
from sklearn.metrics import roc_curve, auc

dtree_auc = 0

def plot_roc(dt_y_true, dt_probs):
    
    # Use sklearn.metrics.roc_curve() to get the values based on what the funciton returns 
    dtree_fpr, dtree_tpr, threshold = roc_curve(dt_y_true, dt_probs)
    
    # Use sklearn.metrics.auc() to get the AUC score of the model 
    dtree_auc_val =  auc(dtree_fpr, dtree_tpr)
    
    # Report the AUC score for the model you have created
    print('AUC=%0.2f'%dtree_auc_val) 
    
    
    # Plot the ROC curve using the probabilities and the true y values as passed into the fuction
    
    plt.plot(dtree_fpr, dtree_tpr, label = 'AUC=%0.2f'%dtree_auc_val, color = 'darkorange')
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'b--')
    plt.xlim([0,1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return dtree_auc_val

dtree_probs = clf.predict_proba(X_test) [:,1]
dtree_auc = plot_roc(dtree_y_true, dtree_probs) # 1.5 (this is where the value is returned by the function and verified)

# Report which cross validation value of k gave the best result as well as the max accuracy score 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 11):

    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=100)

    results_skfold_acc = (cross_val_score(clf, X, y, cv = skfold)).mean() * 100.0
    
    if results_skfold_acc > max_acc: # conditional check for getting max value and corresponding k value
        
        max_acc = results_skfold_acc
        best_k = k

    print("Accuracy: %.2f%%" % (results_skfold_acc))

best_accuracy =  max_acc
best_k_fold =  best_k

print(best_accuracy, best_k_fold)
