import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df['diagnosis'].replace('M', 0, inplace=True)
df['diagnosis'].replace('B', 1, inplace=True)
y=df.diagnosis.values
x_data=df.drop('diagnosis', axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
print('Training set shape:', x_train.shape, y_train.shape)
print('Testing set shape:', x_test.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression
loreg=LogisticRegression(max_iter=200)
loreg.fit(x_train, y_train)
print('Accuracy of model is {}:'.format(loreg.score(x_test, y_test)))

from sklearn.metrics import confusion_matrix
y_true=y_test
y_pred=loreg.predict(x_test)
cm=confusion_matrix(y_true, y_pred)

import seaborn as sns
f, ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, linewidth=1, fmt='.0f', ax=ax)
plt.show()
