import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


# -------------------------------------- Load Data -------------------------------------
data_train = pd.read_csv('1_term/2_Lesson_:_churn/train.csv')
data_test = pd.read_csv('1_term/2_Lesson_:_churn/test.csv')
submission = pd.read_csv('1_term/2_Lesson_:_churn/submission.csv')

target_col = 'Churn'

# ------------------------------------ Preprocessing -----------------------------------

# Пустые значения заменим на NaN
data_train.replace(' ', np.nan, inplace=True)
data_test.replace(' ', np.nan, inplace=True)

# В данном столбце есть NaN, тк пользователь ещё ни разу не платил. Заменим на 0
data_train["TotalSpent"] = data_train.TotalSpent.fillna(0).astype(float)
data_test["TotalSpent"] = data_test.TotalSpent.fillna(0).astype(float)
# Теперь в данных нет столбцов с type == object

#
X_train = data_train.drop(target_col, axis=1)
y_train = data_train[target_col]

X_test = data_test.copy()


# Категориальные признаки
cat_columns = data_train.select_dtypes(include='object').columns

for column in cat_columns:
	# По текущей колонке посчитаем вклад каждого класса в таргет
	data_column = data_train.groupby(column)['Churn'].sum().sort_values()

	# Создадим словарь с новыми значениями классов, в зависимости от их вклада в таргет
	# Первое значение будет иметь значение 0, далее по возрастающей
	new_values = {key: ind for ind, key in enumerate(data_column.index)}

	X_train[column] = X_train[column].replace(new_values)
	X_test[column] = X_test[column].replace(new_values)

# X_train.head(10)



# ---------------------------------------------- RandomForestClassifier ----------------------------------------------



tree_params_rf = {
	'n_estimators': [20],			# range(10, 100, 10)
	'criterion': ['gini'],  		# ['gini', 'entropy']
	'max_depth': [8],  				# range(4, 14, 2)
	'max_features': [3],  			# range(3, 15, 2)
	'min_samples_split': [71],  	# range(10, 100, 10)
	'min_samples_leaf': [16]  		# range(5, 35, 5)
}

clf_rf = RandomForestClassifier(random_state=42)

gridsearch = GridSearchCV(
	estimator=clf_rf,
	param_grid=tree_params_rf,
	cv=3,
	scoring='roc_auc'
)

# обучаем модель на train
random_forest = gridsearch.fit(X_train, y_train)

# print(f'best score RandomForestClassifier: {random_forest.best_score_}')
print(f'best params RandomForestClassifier: {random_forest.best_params_}')


y_pred_proba = random_forest.predict_proba(X_test)[:, 1]

submission['Churn'] = y_pred_proba
# submission.to_csv('submission_rendom_forest.csv', index=False)

