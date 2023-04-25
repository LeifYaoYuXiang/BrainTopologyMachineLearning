import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="BrainTopologyMachineLearning")
    parser.add_argument("--seed", type=int, default=42)
    # data
    parser.add_argument('--dataset_dir', type=str, default=r'data')
    parser.add_argument('--dataset_type', type=str, default=r'EEG')

    # model
    parser.add_argument("--model_name", type=str, default="bagging")

    # train and test
    parser.add_argument("--n_split", type=int, default=5)
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--comment', type=str, default='default comment')
    parser.add_argument('--log_filepath', type=str, default='run')

    args = parser.parse_args()
    return args


