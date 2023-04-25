from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def model_build(model_config: dict):
    model_name = model_config['model_name']
    assert model_name in ['svm', 'bagging', 'decisionTree']
    if model_name == 'svm':
        model = SVC()
    elif model_name == 'bagging':
        model = BaggingClassifier()
    elif model_name == 'decisionTree':
        model = DecisionTreeClassifier()
    else:
        raise NotImplementedError
    return model

