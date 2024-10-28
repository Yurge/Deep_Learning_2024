import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from lightgbm import  LGBMClassifier

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


# Меняем уникальные классы категориальных фичей на цифры 0,1,2,3
cat_columns = data_train.select_dtypes(include='object').columns

for column in cat_columns:
	# По текущей колонке посчитаем вклад каждого класса в таргет. Сортируем по возрастающей
	data_column = data_train.groupby(column)['Churn'].sum().sort_values()

	# Создадим словарь с новыми значениями классов, в зависимости от их вклада в таргет
	# Первое значение будет иметь значение 0, далее по возрастающей
	new_values = {key: ind for ind, key in enumerate(data_column.index)}
	X_train[column] = X_train[column].replace(new_values)
	X_test[column] = X_test[column].replace(new_values)

# X_train.head(10)


# ---------------------------------------------- LGBMClassifier ----------------------------------------------

params = {
	'class_weight': ['balanced'],  	# [None, 'balanced']
	'boosting_type': ['goss'],  	# ['gbdt', 'goss', 'dart']
	'learning_rate': [0.015],  		# [0.011, 0.015, 0.02, 0.03]
	'colsample_bytree': [0.7],  	# [0.6, 0.7, 0.8]
	'n_estimators': [200],  		# [150, 200, 250]
	'max_depth': [3],  				# [3, 4, 5]
	'subsample': [0.7],  			# [0,6, 0.7, 0.8]
	'min_child_samples': [11]  		# [9, 10, 11]
}

model_lgbm = LGBMClassifier(random_state=42, metric='auc')

grid_search = GridSearchCV(model_lgbm,
						   param_grid=params,
						   cv=5,
						   n_jobs=-1,
						   scoring='roc_auc')

lgbm = grid_search.fit(X_train, y_train)

best_params = lgbm.best_estimator_
print(best_params)

y_test_pred_proba = lgbm.predict_proba(X_test)[:, 1]

submission['Churn'] = y_test_pred_proba
submission.to_csv('submission_LGBM.csv', index=False)      # ROC-AUC вашего решения равен