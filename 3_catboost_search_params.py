import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
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


# ---------------------------------------------- CatBoostClassifier ----------------------------------------------

params = {'iterations': [750],  				    	# [500, 800, 1000, 1200]
		'learning_rate': [0.009],      					# [0.005, 0.009, 0.01, 0.02]
    	'depth': [3],                        			# [4, 6, 8]
        'l2_leaf_reg': [4],               				# [1, 3, 5, 7, 9]
        'grow_policy' : ['Depthwise']					# ['Depthwise', 'SymmetricTree', 'Lossguide']
        }


model_catboost = CatBoostClassifier(verbose=False, custom_metric='AUC', random_state=42)

grid_search = GridSearchCV(model_catboost, params, cv=5, scoring='roc_auc', verbose=1000)
catboost = grid_search.fit(X_train, y_train)

print(catboost.best_params_)
print('----------------------------------------------------------------------')


y_test_pred_proba = catboost.predict_proba(X_test)[:, 1]

submission['Churn'] = y_test_pred_proba
submission.to_csv('submission_catboost.csv', index=False)      # ROC-AUC вашего решения равен








import joblib

# Deploy the best Grid Search model (you can also deploy the best Random Search model similarly)
# best_model = grid_search.best_estimator_
# best_model = catboost

# Save the best model to a file using joblib
model_filename = "best_catboost_model.joblib"
# joblib.dump(best_model, model_filename)

# Later, to load and use the saved model for predictions:
# loaded_model = joblib.load(model_filename)

# Make predictions using the loaded model
y_pred_loaded = loaded_model.predict(X_test)

# Print classification report for the loaded model
print("Loaded Model - Classification Report:")
#print(classification_report(y_test, y_pred_loaded))