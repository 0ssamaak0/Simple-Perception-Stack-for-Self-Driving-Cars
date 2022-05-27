import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV


def SVM(car_features, notcar_features):
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # we fit on training only and apply on both, the test must be completely uknown
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # # Linear SVM
    # svc = LinearSVC()
    # svc.fit(scaled_X, y)

    # nonLinear
    svc = SVC()
    params = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = GridSearchCV(svc, params)
    svc.fit(scaled_X, y)

    return svc, X_scaler
