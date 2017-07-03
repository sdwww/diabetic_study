from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

import DataProcess


def random_forest(x_train, y_train, x_test, y_test, n_trees=10):
    rfc = RandomForestClassifier(n_estimators=n_trees)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    y_pred_prob = rfc.predict_proba(x_test)
    print('精确度：', accuracy_score(y_test, y_pred))
    print('准确度：', precision_score(y_test, y_pred))
    print('召回率：', recall_score(y_test, y_pred))
    print('AUC：', roc_auc_score(y_test, y_pred_prob[:,1]))


if __name__ == '__main__':
    data_set = DataProcess.word2number()
    random_forest(data_set[0], data_set[1], data_set[2], data_set[3])
