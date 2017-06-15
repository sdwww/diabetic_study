from sklearn.ensemble import RandomForestClassifier


def random_forest(x_train, y_train, x_test, y_test, n_trees=10):
    rfc = RandomForestClassifier(n_estimators=n_trees)
    rfc.fit(x_train, y_train)
    accuracy_score = rfc.score(x_test, y_test)
    return accuracy_score, rfc.feature_importances_