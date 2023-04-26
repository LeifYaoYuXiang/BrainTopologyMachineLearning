from utils import read_data_from_pickle
import os.path
import dgl
import numpy as np
import torch


# 数据预处理，获得可以训练的机器学习数据
def preprocess_data(dataset_config):
    dataset_dir = dataset_config['dataset_dir']
    dataset_type = dataset_config['dataset_type']
    label_filepath = dataset_config['label_filepath']
    label_list = read_data_from_pickle(label_filepath)
    assert dataset_type in ['fMRI', 'EEG']
    # 获取所有的文件地址
    dataset_dir = os.path.join(dataset_dir, dataset_type)
    total_filename = os.listdir(dataset_dir)
    tmp = []
    for each_filename in total_filename:
        each_filename = os.path.join(dataset_dir, each_filename)
        tmp.append(each_filename)
    total_filename = tmp
    # fMRI同构图的编码
    if dataset_type == 'fMRI':
        i = 0
        for each_filename in total_filename:
            graph: dgl.graph = read_data_from_pickle(each_filename)
            node_data: torch.Tensor = graph.ndata['feat']
            torch.reshape(node_data, (-1,))
            node_data = node_data.numpy()
            if i == 0:
                encoded_data = node_data
            else:
                encoded_data = np.vstack((encoded_data, node_data))
            i = i + 1
        return encoded_data, label_list
    # EEG异构图的编码
    else:
        i = 0
        for each_filename in total_filename:
            graph: dgl.heterograph = read_data_from_pickle(each_filename)
            left_node_data: torch.Tensor = graph.nodes['l'].data['feat']
            right_node_feat: torch.Tensor = graph.nodes['r'].data['feat']
            node_feat = torch.concat((left_node_data, right_node_feat), dim=0)
            torch.reshape(node_data, (-1,))
            node_data = node_data.numpy()
            if i == 0:
                encoded_data = node_data
            else:
                encoded_data = np.vstack((encoded_data, node_data))
            i = i + 1
        return encoded_data, label_list
