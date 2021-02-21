```python

#Import libraries
from sklearn.metrics import confusion_matrix

#Dummy Data
#Actual values
y_actual = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

#Predicted values
y_predicted = [0, 0, 1, 1, 1, 0, 1, 1, 1, 1]


#As per above dummy data expected tp=3,fn=1,fp=4,tn=2
#--------------------------------------------------------------------------
#Case1 - Predicted on rows and Actual on columns with 0,1 labels
#Display of matrix - Correct display
#Extracted values - Correct extracted values

cm = confusion_matrix(y_predicted, y_actual, labels=[0, 1])
print(cm)

#Extract values
print("tp", (cm[1][1])) # Expected 3
print("fp", (cm[1][0])) # Expected 4
print("fn", (cm[0][1])) # Expected 1
print("tn", (cm[0][0])) # Expected 2

#--------------------------------------------------------------------------

#Case2 - Actual on rows and Predicted on columns with 0,1 labels
#Display of matrix - Correct display
#Extracted values - Correct extracted values

cm = confusion_matrix(y_actual, y_predicted, labels=[0, 1])
print(cm)


#Extract values
print("tp", (cm[1][1])) # Expected 3
print("fn", (cm[1][0])) # Expected 1
print("fp", (cm[0][1])) # Expected 4
print("tn", (cm[0][0])) # Expected 2

#--------------------------------------------------------------------------

#Case3 - Predicted on rows and Actual on columns with 1,0 labels
#Display of matrix - Correct display
#Extracted values - Extracted values are INCORRECT by confusion matrix

cm = confusion_matrix(y_predicted, y_actual, labels=[1, 0])
print(cm)


#Extract values
print("tp", (cm[1][1])) # Expected 3
print("fp", (cm[1][0])) # Expected 4
print("fn", (cm[0][1])) # Expected 1
print("tn", (cm[0][0])) # Expected 2

#--------------------------------------------------------------------------

#Case4 - Actual on rows and Predicted on columns with 1,0 labels
#Display - Correct display
#Extracted values - Extracted values are INCORRECT by confusion matrix

cm = confusion_matrix(y_actual, y_predicted, labels=[1, 0])
print(cm)

#Extract values
print("tp", (cm[1][1])) # Expected 3
print("fn", (cm[1][0])) # Expected 1
print("fp", (cm[0][1])) # Expected 4
print("tn", (cm[0][0])) # Expected 2

```

```bash

import sklearn; sklearn.show_versions()
System:
    python: 3.9.1 (v3.9.1:1e5d33e9b9, Dec  7 2020, 12:10:52)  [Clang 6.0 (clang-600.0.57)]
executable: /Users/sylviachadha/Desktop/Tools/PyCharm/venv/bin/python
   machine: macOS-10.16-x86_64-i386-64bit
Python dependencies:
          pip: 21.0.1
   setuptools: 53.0.0
      sklearn: 0.24.1
        numpy: 1.19.5
        scipy: 1.5.4
       Cython: None
       pandas: 1.1.5
   matplotlib: 3.3.4
       joblib: 1.0.1
threadpoolctl: 2.1.0
Built with OpenMP: True

```
