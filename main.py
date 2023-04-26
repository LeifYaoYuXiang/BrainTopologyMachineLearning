import os.path
import joblib
from argument_parser_v1 import parser_args
from data_preprocess_v1 import preprocess_data
from train_test_eval_v1 import train_test_eval_v1
from utils import get_summary_writer, record_configuration


def main(args):
    log_filepath = args.log_filepath
    summary_writer, log_dir = get_summary_writer(log_filepath)

    dataset_config = {
        'dataset_dir': args.dataset_dir,
        'dataset_type': args.dataset_type,
        'label_filepath': args.label_filepath,
    }
    model_config = {
        'model_name': args.model_name,
    }
    train_test_config = {
        'n_split': args.n_split,
        'performance_record_filepath': os.path.join(log_dir, 'record.txt'),
        'test_ratio': args.test_ratio,
        'comment': args.comment,
    }
    record_configuration(save_dir=log_dir, configuration_dict={
        'MODEL': model_config,
        'DATASET': dataset_config,
        'TRAIN': train_test_config,
    })
    print(model_config, '\n', dataset_config, '\n', train_test_config, '\n')
    feature_x, label_y = preprocess_data(dataset_config)
    model_list = train_test_eval_v1(model_config, feature_x, label_y, train_test_config)
    for i in range(len(model_list)):
        each_model = model_list[i]
        joblib.dump(each_model, os.path.join(log_dir, str(i)+'.pkl'))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = parser_args()
    main(args)
