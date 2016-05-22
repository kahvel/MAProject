from sklearn.svm import LinearSVC


class LinearSVM(LinearSVC):
    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000):
        LinearSVC.__init__(self, penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling, class_weight, verbose, random_state, max_iter)

    # def predict_proba(self, X):
    #     return self.decision_function(X)
