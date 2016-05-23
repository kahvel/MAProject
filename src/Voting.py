from sklearn.ensemble import VotingClassifier


class Voting(VotingClassifier):
    def __init__(self, estimators, voting="hard", weights=None):
        VotingClassifier.__init__(self, estimators, voting, weights)
        self.estimators_ = [estimator[1] for estimator in estimators]
