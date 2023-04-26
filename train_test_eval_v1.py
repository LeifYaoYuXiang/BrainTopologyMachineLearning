from sklearn.model_selection import train_test_split
from metric import ClassificationMetric
from model_build_v1 import model_build
from utils import write_list_to_file


def train_test_eval_v1(model_config, feature_x, label_y, train_test_config):
    n_split = train_test_config['n_split']
    test_ratio = train_test_config['test_ratio']
    performance_record = []
    performance_record_filepath = train_test_config['performance_record_filepath']
    model_list = []
    # 5-split cross validation
    for each_split in range(n_split):
        print('Fold ' + str(each_split))
        model = model_build(model_config)
        x_train, x_test, y_train, y_test = train_test_split(feature_x, label_y, test_size=test_ratio)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc, precision, recall, f1 = ClassificationMetric.score(y_test, y_pred)
        print(acc, precision, recall, f1)
        performance_record.append('split: {}, Acc: {}, Precision: {} Recall: {}, F1: {} '.format(each_split, acc, precision, recall, f1))
        model_list.append(model)
    write_list_to_file(performance_record_filepath, performance_record)
    return model_list



