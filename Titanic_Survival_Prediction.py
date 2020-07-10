import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


PATH = '/kaggle/input/titanic'
PATH2 = '/kaggle/input/testans'
test = pd.read_csv(f'{PATH}/test.csv')
train = pd.read_csv(f'{PATH}/train.csv')
combine = train.append(test,ignore_index=True, sort=False)
gender_submission = pd.read_csv(f'{PATH}/gender_submission.csv')


train

train.info()

train.describe()

train.describe(include=['O'])

train.corr()

combine=combine.drop(['Ticket'], axis=1)

combine[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

combine[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

combine.replace(to_replace ="male", value =1, inplace=True)
combine.replace(to_replace ="female", value =2, inplace=True)

combine.head()

combine['AgeBand'] = pd.cut(combine['Age'], 5)
combine['AgeBand'].groupby(combine['AgeBand']).count()

combine.isna().sum()

combine['Age'].fillna(0,inplace=True)

AgeG=[]
for value in combine['Age']:
    value = int(value)
    if value<=16:
        AgeG.append(0)
    elif (value>16) & (value<=25):
        AgeG.append(1)
    elif (value>25) & (value<=50):
        AgeG.append(2)
    elif (value>50) & (value<=65):
        AgeG.append(3)
    else:
        AgeG.append(4)
combine['Age'] = AgeG

combine.drop(['AgeBand'], axis=1, inplace=True)

# Determining whether person is alone or not
for data in combine:
    combine['FamilySize'] = combine['SibSp'] + combine['Parch'] + 1
# combine[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for data in combine:
    combine['Alone'] = (combine['FamilySize'] == 1).astype(int)

combine[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean().sort_values(by='Survived', ascending=False)


for data in combine:
    combine['Age*Class'] = combine.Age * combine.Pclass

combine.replace(to_replace ="C", value =1,inplace=True)
combine.replace(to_replace ="Q", value =2,inplace=True)
combine.replace(to_replace ="S", value =3,inplace=True)

combine['Embarked'].fillna(0,inplace=True)
combine['Fare'].fillna(combine['Fare'].median(),inplace=True)

FareG=[]
for value in combine['Fare']:
    value = int(value)
    if value<=8:
        FareG.append(0)
    elif (value>8) & (value<=15):
        FareG.append(1)
    elif (value>15) & (value<=32):
        FareG.append(2)
    elif (value>32):
        FareG.append(3)
combine['Fare'] = FareG

combine.Cabin.fillna('0', inplace=True)
combine.loc[combine.Cabin.str[0] == 'A', 'Cabin'] = 1
combine.loc[combine.Cabin.str[0] == 'B', 'Cabin'] = 2
combine.loc[combine.Cabin.str[0] == 'C', 'Cabin'] = 3
combine.loc[combine.Cabin.str[0] == 'D', 'Cabin'] = 4
combine.loc[combine.Cabin.str[0] == 'E', 'Cabin'] = 5
combine.loc[combine.Cabin.str[0] == 'F', 'Cabin'] = 6
combine.loc[combine.Cabin.str[0] == 'G', 'Cabin'] = 7
combine.loc[combine.Cabin.str[0] == 'T', 'Cabin'] = 8

convert = {'Cabin': int}
combine = combine.astype(convert)

for dataset in combine:
    combine['Title'] = combine.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(combine['Title'], combine['Sex'])

for dataset in combine:
    combine['Title'] = combine['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    combine['Title'] = combine['Title'].replace('Mlle', 'Miss')
    combine['Title'] = combine['Title'].replace('Ms', 'Miss')
    combine['Title'] = combine['Title'].replace('Mme', 'Mrs')

combine[['Title', 'Survived']].groupby(combine['Title'], as_index=False).mean()


title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
combine.Title = [title_mapping[item] for item in combine.Title]
#    combine['Title'] = combine['Title'].fillna(0)


for data in combine:
    combine['Fare*Title'] = combine.Fare * combine.Title


#X_train  = df_data.loc[~pd.isnull(df_data.Survived),['Age', 'Sex', 'Pclass']]
X_train = combine.loc[pd.notnull(combine.Survived), ['Pclass','Sex','Age','FamilySize','Alone','Fare','Embarked','Cabin','Title','Fare*Title']]
Y_train = combine.loc[pd.notnull(combine.Survived), ['Survived']]
X_test = combine.loc[pd.isnull(combine.Survived), ['Pclass','Sex','Age','FamilySize','Alone','Fare','Embarked','Cabin','Title','Fare*Title']]
Y_test = []

convert = {'Survived': int}
Y_train = Y_train.astype(convert)

X_test.isna().sum()


LR = LogisticRegression()
LR.fit(X_train, Y_train)
Y_predictlr = LR.predict(X_test)
LR_error = round(LR.score(X_train, Y_train) * 100, 2)
LR_error

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_predictlr})
submission.to_csv('submission0.csv', index=False)