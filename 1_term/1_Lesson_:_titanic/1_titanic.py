import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score




data = pd.read_csv('game_of_thrones_train.csv', index_col='S.No')
data_test = pd.read_csv('game_of_thrones_test.csv', index_col='S.No')

# print(data.columns)
# print(data.head(4))
# print(data.info())
# data.isna().sum()

df = data.copy()

def fill_nan(column, dts):
    if dts[column].dtype in ['O', 'object']:
        dts.fillna({column: 'Other'}, inplace=True)
    elif dts[column].dtype in ['int64', 'float64']:
        dts.fillna({column: dts[column].mean()}, inplace=True)

for col in df.columns:
    fill_nan(col, df)

# print(df.head(5))


def popular(data_frame):
    if data_frame['popularity'] > 0.5:
        return 1
    else:
        return 0


def booldead(data_frame):
    if data_frame['numDeadRelations'] > 0:
        return 1
    else:
        return 0


df['isPopular'] = df.apply(popular, axis=1)
df['boolDeadRelations'] = df.apply(booldead, axis=1)

# df[['popularity', 'isPopular', 'numDeadRelations', 'boolDeadRelations']].head(10)
# print(df.info())

# print(df.culture.nunique())  # 52

# Подсказка
cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
}


for key, value in cult.items():
    df['culture'] = df['culture'].replace(value, key)

df.drop(columns=['popularity', 'numDeadRelations'], inplace = True)
#print(df.info())

# plt.figure(figsize = (4, 4))
# df['isAlive'].hist(density=False, bins=20)
# plt.ylabel('count')
# plt.xlabel('isAlive')
# plt.show()

obj_cols = df.select_dtypes(include=['object'])
for col in obj_cols.columns:
    df[col] = df[col].astype('category')
    df[f'{col}_cat'] = df[col].cat.codes

# print(df.info())

corr_columns = df.select_dtypes(exclude=['category']).columns.tolist()
# f, ax = plt.subplots(figsize=(23, 12))
# sns.heatmap(df[corr_columns].corr(), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7})
# plt.show()

train_columns = df.select_dtypes(exclude=['category']).columns.tolist()
X = df[train_columns].drop(columns=['isAlive'])
y = df['isAlive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)


def fit_pred_accur(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f'{model} \nAccuracy : {accuracy:.4f}'


# Шаг 1. создание модели
models = {
    'logistic_regression': LogisticRegression(solver='liblinear'),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'RandomForest': RandomForestClassifier(max_depth=6, n_estimators=15),
    'GaussianProcess': GaussianProcessClassifier(),
    'GaussianNB': GaussianNB(),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTree': DecisionTreeClassifier()
}

for model in models:
    print(fit_pred_accur(models[model]))


best_model = models['RandomForest']
best_model.fit(X_train, y_train)

submission = pd.read_csv("submission.csv", index_col='S.No')


dt = data_test.copy()

# заполняет пустые ячейки
for col in dt.columns:
    fill_nan(col, dt)

# создадим две новые колонки
dt['isPopular'] = dt.apply(popular, axis=1)
dt['boolDeadRelations'] = dt.apply(booldead, axis=1)

# удалим ненужные колонки
dt.drop(columns=['popularity', 'numDeadRelations'], inplace = True)

# упростим колонку culture, объединив схожие названия
for key, value in cult.items():
    dt['culture'] = dt['culture'].replace(value, key)

#
obj_cols = dt.select_dtypes(include=['object'])
for col in obj_cols.columns:
    dt[col] = dt[col].astype('category')
    dt[f'{col}_cat'] = dt[col].cat.codes

# print(dt.head())

columns = dt.select_dtypes(exclude=['category']).columns.to_list()
y_pred_new = best_model.predict(dt[columns])

submission['isAlive'] = y_pred_new
# print(submission)

# submission.to_csv("/content/new_submission.csv", index=False)