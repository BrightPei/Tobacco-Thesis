from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
begin = time.time()

features = ['B', 'G', 'R', 'H', 'S', 'a*', 'b*', 'dry_up', 'wet_up', 'baketime',
       'driage', 'waterloss_rate', 'yel_per', 'contrast', 'homogeneity',
       'energy']
dataset = pd.read_csv(r"F:\pythoncode\Tobacco_thesis\Data_set\All_Data\200812192633-SY-B-1-C-87.csv")
#将特征数据和标签分开
X,y = dataset.drop(columns=["label","holdingtime","Unnamed: 0"]),dataset.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=666)
y_test = y_test.values.ravel()
y_train = y_train.values.ravel()
xgb = XGBClassifier(n_estimators=100,
                    objective='multi:softmax', booster='gbtree',
                    random_state=567, seed=567,eval_metric='mlogloss')
# 网格暴力搜索的参数值
tuned_params = [
    {
        'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': range(1, 10, 1),
        'subsample': [i / 10.0 for i in range(5, 11)],
        'colsample_bytree': [i / 10.0 for i in range(5, 11)],
    }
    ]

scores = ["precision", "recall"]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(estimator=xgb, param_grid=tuned_params, scoring="%s_macro" % score, n_jobs=4, refit=True,
                       cv=5, verbose=0, pre_dispatch='2*n_jobs', return_train_score=False)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

print('time cost = {0} min'.format(round((time.time() - begin) / 60.0, 2)))
