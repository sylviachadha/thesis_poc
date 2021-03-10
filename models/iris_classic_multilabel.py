# IRIS DATASET - MULTI CLASS PROBLEM
# (Setosa, Versicolor, Virginica)
# EVALUATE DIFFERENT ALGORITHMS ON TOY DATA
#---------------------------------------------------------------------


# STEP 1 Import sklearn and load dataset
#-------------------------------------------
from sklearn.datasets import load_iris
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

iris_bunch = load_iris()
# The data structure is bunch and loads data, feature_names, target, target_names etc

# To see X or features iris_bunch.data as ndarray
print(iris_bunch.data)
print(iris_bunch.data.shape)
print("feature_names", iris_bunch.feature_names)

# To see corresponding labels / target Y
print(iris_bunch.target)
print(iris_bunch.target.shape)
print("target_names", iris_bunch.target_names)

# summarize class distribution
counter = Counter(iris_bunch.target)
print(counter)

# Notation in standard X and y

X = iris_bunch.data
y = iris_bunch.target


# STEP 2 - Split into Train & Test to feed to ML Algorithm
#----------------------------------------------------------
# Check Shape of data and target/label
print("Shape of data", X.shape)
print("Shape of label", y.shape)

# Split into Train and Test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4,stratify=y)

# print shapes of new X objects
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

# print shapes of new y objects
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)


# STEP 3 - Modelling (1.Import, 2.Instantiate, 3.Fit and 4.Predict)
#------------------------------------------------------------------

# Model 1 - Logistic Regression
#------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression # Import
logreg = LogisticRegression(max_iter=200) # Instantiate
logreg.fit(X_train,y_train) # Fit
y_pred_lr = logreg.predict(X_test) # Predict

# Model 2 - KNN
#------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier # Import
knn = KNeighborsClassifier(n_neighbors=5) # Instantiate
knn.fit(X_train,y_train) # Fit
y_pred_knn = knn.predict(X_test) # Predict

# Model 3 - Decision Tree
#------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier # Import
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # Instantiate
dt.fit(X_train, y_train) # Fit
y_pred_dt = dt.predict(X_test) # Predict

# Model 4 - Random Forest
#------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier # Import
rf = RandomForestClassifier(n_estimators=200) # Instantiate
rf.fit(X_train, y_train) # Fit
y_pred_rf = rf.predict(X_test) # Predict


# Model 5 - SVM - Linear
#------------------------------------------------------------------

from sklearn.svm import SVC  # Import
svc = SVC()  # Instantiate
svc.fit(X_train, y_train)  # Fit
y_pred_svc = svc.predict(X_test)  # Predict

# Model 6 - SVM - Non-Linear
#------------------------------------------------------------------

from sklearn.svm import SVC  # Import
svc_rbf = SVC(kernel='rbf', gamma=1.0)  # Instantiate
svc_rbf.fit(X_train, y_train)  # Fit
y_pred_svcrbf = svc_rbf.predict(X_test)  # Predict

# Model 7 - Naive Bayes
#------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB  # Import
nbayes = GaussianNB()  # Instantiate
nbayes.fit(X_train, y_train)  # Fit (features, labels)
y_pred_nb = nbayes.predict(X_test)  # Predict


# STEP 4 - Evaluation Metrics
#------------------------------------------------------------------

def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred,average='macro')
    precision = precision_score(y_actual, y_pred,average='macro')
    f1score = f1_score(y_actual, y_pred,average='macro')
    return(cm, acc, recall, precision, f1score)

# Model 1 Logistic Regression

lr_result = evaluate_model(y_test, y_pred_lr)
print("Logistic Regression values for cm, accuracy, recall, precision and f1 score", lr_result)

# Model 2 - KNN

knn_result = evaluate_model(y_test, y_pred_knn)
print("KNN values for cm, accuracy, recall, precision and f1 score", knn_result)

# Model 3 - Decision Tree

dt_result = evaluate_model(y_test, y_pred_dt)
print("Decision Tree values for cm, accuracy, recall, precision and f1 score", dt_result)

# Model 4 - Random Forest

rf_result = evaluate_model(y_test, y_pred_rf)
print("Random Forest values for cm, accuracy, recall, precision and f1 score", rf_result)

# Model 5 - SVM - Linear

svclinear_result = evaluate_model(y_test, y_pred_svc)
print("SVM Linear values for cm, accuracy, recall, precision and f1 score", svclinear_result)

# Model 6 - SVM - Non-Linear

svcrbf_result = evaluate_model(y_test, y_pred_svcrbf)
print("SVM RBF Kernel values for cm, accuracy, recall, precision and f1 score", svcrbf_result)

# Model 7 - Naive Bayes

nb_result = evaluate_model(y_test, y_pred_nb)
print("Naive Bayes values for cm, accuracy, recall, precision and f1 score", nb_result)
