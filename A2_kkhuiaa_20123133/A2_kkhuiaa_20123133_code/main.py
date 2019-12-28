#%%
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
from sklearn.metrics import make_scorer, accuracy_score
from xgboost import XGBClassifier

import categorytoordinal
#%%
#Creating dummy variables for categorical variable is not used here, there are few drawbacks:
# 1. curse of dimensionality
# 2. too sparse to train
# Instead we will transform the categorical variable into ordianal variable (1 ,2 ,3, 4...), where this number score is ranked by its mean of the y (target).

X_raw = pd.read_csv('trainFeatures.csv')
X = X_raw.drop('education-num', axis=1) #drop it since we will handle the categorical variable
Y = pd.read_csv('trainLabels.csv', header=None, names=['target'])

X_numeric = X.select_dtypes(include=['number'])

X_categorical = X[[col for col in X.columns.tolist() if col not in X_numeric.columns]]
categorical_cols = X_categorical.columns.tolist() #record it for test dataset

for col in X_categorical:
    X_categorical[col] = X_categorical[col].str.strip()

cto = categorytoordinal.CategoryToOrdinal(other_threshold=30)
cto.fit(X_categorical, Y['target'])
X_categorical_transformed = cto.transform(X_categorical)

imp = Imputer(missing_values=np.nan, strategy='mean') #even if there is no np.nan, we still do the na value treatment, since testing dataset may contains NA value.
imp.fit(X_numeric)
X_numeric_fillna = cto.transform(X_numeric)

pca = decomposition.PCA(n_components=.95)
pca.fit(X_numeric_fillna)
X_numeric_pca = pd.DataFrame(data=pca.transform(X_numeric_fillna), columns=['pca_'+str(i+1) for i in range(pca.n_components_)], index=X_numeric_fillna.index)
numeric_cols = X_numeric_pca.columns.tolist() #record it for test dataset

X_merged = pd.concat([X_numeric_pca, X_categorical_transformed], axis=1)

sc = StandardScaler()
sc.fit(X_merged)
X_merged = pd.DataFrame(data=sc.transform(X_merged), index=X_merged.index, columns=X_merged.columns)
X_merged = X_merged[numeric_cols+categorical_cols]

print(X_merged.shape) #(34189, 9)

random_state = 20
param = [{
    'learning_rate': [.01]
    , 'subsample': [.45, .5, .55, .6]
    # , 'subsample': [.7, .75, .8, .85, .9]
    , 'n_estimators': [1000]
    , 'min_child_weight': [10]
    , 'scale_pos_weight': [1.2]
    , 'reg_lambda': [.1, .01, .05]
}]
model = XGBClassifier(random_state=random_state, n_jobs=-1)

gridsearch = GridSearchCV(model, param_grid=param, cv=10, n_jobs=-1, scoring=make_scorer(accuracy_score))
t0 = time.time()
gridsearch.fit(X_merged, Y['target'])
t1 = time.time()

print('time:', round(t1-t0, 2)) #calculate the time for gridsearch
#time: 297.84s
print('best params:', gridsearch.best_params_)
# best params: {'learning_rate': 0.01, 'min_child_weight': 10, 'n_estimators': 1000, 'reg_lambda': 0.01, 'scale_pos_weight': 1.2, 'subsample': 0.5}
best_position = gridsearch.best_index_
print('best train score:', gridsearch.cv_results_['mean_train_score'][best_position])
# best train score: 0.8369423597821433
print('best test score:', gridsearch.cv_results_['mean_test_score'][best_position])
# best test score: 0.8348591652285823

#predict the test data
#basically we just apply the transformation above to unseen data
X_test_raw = pd.read_csv('testFeatures.csv')
X_test = X_test_raw.drop('education-num', axis=1)
X_test_numeric = X_test.select_dtypes(include=['number'])
X_test_categorical = X_test[categorical_cols]

for col in X_categorical:
    X_test_categorical[col] = X_test_categorical[col].str.strip()

X_test_categorical_transformed = cto.transform(X_test_categorical)
X_test_numeric_fillna = cto.transform(X_test_numeric)

X_test_numeric_pca = pd.DataFrame(data=pca.transform(X_test_numeric_fillna), columns=['pca_'+str(i+1) for i in range(pca.n_components_)], index=X_test_numeric_fillna.index)

X_test_merged = pd.concat([X_test_numeric_pca, X_test_categorical_transformed], axis=1)

X_test_merged = pd.DataFrame(data=sc.transform(X_test_merged), index=X_test_merged.index, columns=X_test_merged.columns)
X_test_merged = X_test_merged[numeric_cols+categorical_cols]

print(X_test_merged.shape)

y = pd.Series(gridsearch.best_estimator_.predict(X_test_merged))
y.to_csv('A2_kkhuiaa_20123133_prediction.csv', index=False, header=False)
