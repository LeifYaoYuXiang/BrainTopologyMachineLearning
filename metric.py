from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class ClassificationMetric:
    @staticmethod
    def score(y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        micro_precision = precision_score(y_test, y_pred, average='micro')
        micro_recall = recall_score(y_test, y_pred, average='micro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        return round(acc, 4), round(micro_precision, 4), round(micro_recall, 4), round(micro_f1, 4)
