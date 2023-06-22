# Import Library
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from IPython import get_ipython 
import warnings 
warnings.filterwarnings ("ignore")
import joblib

data = pd.read_csv('hr_data.csv')
data.duplicated().sum()
data.drop_duplicates()
data = data.rename(columns = {'sales':'department'})

data['department'] = np.where(data['department'] == 'support',
                              'technical', data['department'])
data['department'] = np.where(data['department'] == 'IT',
                              'technical', data['department'])
cat_vars = ['department','salary']
for var in cat_vars:
  cat_list = 'var' +  '_' + var
  cat_list = pd.get_dummies(data[var], prefix = var)
  data1 = data.join(cat_list)
  data = data1

data.drop(data.columns[[8,9]], axis=1, inplace=True)
data.columns.values

hr_vars = data.columns.values.tolist()
y = ['left']
x = [i for i in hr_vars if i not in y]


cols = ['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'department_RandD', 'department_hr', 'department_management', 'salary_high']
X = data[cols]
y = data['left']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
logreg = LogisticRegression ()
logreg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = logreg.predict(X_test)
print('Logistic Regression Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Random Forest Accuracy: {:.3f}".format(accuracy_score(y_test, rf.predict(X_test))))
# Save the trained model as model.pkl
joblib.dump(rf, 'model.pkl')