import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)
#enc = OneHotEncoder(handle_unknown='ignore')
#X = enc.fit_transform(X)  # This right here

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print("RandomForest Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))

from autosklearn.classification import AutoSklearnClassifier

#automl = AutoSklearnClassifier(time_left_for_this_task=300, resampling_strategy="cv", resampling_strategy_arguments={"folds":10})
automl = AutoSklearnClassifier(time_left_for_this_task=900, include = {'classifier': ["random_forest"]}, exclude=None)
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
overfit = automl.predict(X_train)
print("AutoML Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))
print("AutoML Train Accuracy", sklearn.metrics.accuracy_score(y_train, overfit))

