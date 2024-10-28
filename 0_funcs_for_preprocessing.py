import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer


df = pd.read_csv('1_term/1_Lesson_:_titanic/game_of_thrones_train.csv')

#
def fill_nan(column, dts):
    if dts[column].dtype in ['O', 'object']:
        dts.fillna({column: 'Other'}, inplace=True)
    elif dts[column].dtype in ['int64', 'float64']:
        dts.fillna({column: dts[column].mean()}, inplace=True)


for col in df.columns:
    fill_nan(col, df)



# Алгоритм перевода категориальных признаков в числовые (с созданием новой колонки) при помощи cat.codes
def columns_cat_code(data):
    obj_cols = data.select_dtypes(include=['object'])
    for col in obj_cols.columns:
        data[col] = data[col].astype('category')
        data[f'{col}_cat'] = data[col].cat.codes


# Выбрать колонки с тимом данных != category
def select_columns_without_type(data, type):
    columns = df.select_dtypes(exclude=[type]).columns.to_list()
    return columns


# Меняем уникальные классы категориальных фичей на цифры 0,1,2,3... в зависимости от их вклада в таргет
cat_columns = data_train.select_dtypes(include='object').columns
for column in cat_columns:
	# По текущей колонке посчитаем вклад каждого класса в таргет. Сортируем по возрастающей
	data_column = data_train.groupby(column)['Churn'].sum().sort_values()
	# Создадим словарь с новыми значениями классов, в зависимости от их вклада в таргет
	# Первое значение будет иметь значение 0, далее по возрастающей
	new_values = {key: ind for ind, key in enumerate(data_column.index)}
	X_train[column] = X_train[column].map(new_values)
	X_test[column] = X_test[column].map(new_values)

# X_train.head(10)


#
def popular(data_frame):
    if data_frame['popularity'] > 0.5:
        return 1
    else:
        return 0


df['isPopular'] = df.apply(popular, axis=1)


#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#
cult = {'Summer Islands': ['summer islands', 'summer islander', 'summer isles']}
for key, value in cult.items():
    df['culture'] = df['culture'].replace(value, key)



# удалим ненужные колонки
def drop_columns(data, cols):
    data.drop(columns=cols, inplace = True)


drop_columns(df, ['popularity', 'numDeadRelations', 'name'])






# График heatmap
# corr_columns = df.select_dtypes(exclude=['category']).columns.tolist()
# f, ax = plt.subplots(figsize=(23, 12))
# sns.heatmap(df[corr_columns].corr(), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7})
# plt.show()



train_columns = df.select_dtypes(exclude=['category']).columns.tolist()
X = df[train_columns].drop(columns=['isAlive'])
y = df['isAlive']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42,
                                                    stratify=y)


def fit_pred_accur(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f'{model} \nAccuracy : {accuracy:.4f}'


# Шаг 1. создание модели
models = {
    'logistic_regression': LogisticRegression(),
    'AdaBoost': AdaBoostClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GaussianProcess': GaussianProcessClassifier(),
    'GaussianNB': GaussianNB(),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTree': DecisionTreeClassifier()
}

# for model in models:
#     print(fit_pred_accur(models[model]))
# print('-------------------------------------------------------------')


# Функция для определения метрик модели
def metrics(model, X_test=X_test, y_test=y_test):
	y_pred = model.predict(X_test)
	y_pred_prob = model.predict_proba(X_test)[:, 1]

	accuracy = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	roc_auc = roc_auc_score(y_test, y_pred_prob)

	print(f'accuracy : {accuracy}\n'
		  f'f1 : {f1}\n'
		  f'roc_auc : {roc_auc}')

	# return roc_auc


# Посмотрим на значимость параметров в деревьях, т.е. как часто используется параметр при сплите
best_model = DecisionTreeClassifier()
imp = pd.DataFrame(best_model.feature_importances_, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh')
# plt.show()



# колонки с коэфф P > 0.005 скорее всего можно удалить
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import statsmodels.stats.tests.test_influence

model_sm = sm.GLM(
    df["isAlive"],
    df[['dateOfBirth', 'book4', 'age', 'culture_cat']],
    family=families.Binomial(),
).fit()
print(model_sm.summary())


# Соберем колонки с числовыми значениями и посмотрим в каждой из них есть ли экстремальные выбросы
for column in X.select_dtypes(exclude=['object']):
    print(df[column].sort_values())
    print('-------------------------------')


# Наивный алгоритм Нормализации
def norm(data_column):
    return (data_column - min(data_column)) / (max(data_column) - min(data_column))
