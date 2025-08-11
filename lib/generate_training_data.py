import os
import csv
import pickle
import argparse
import numpy as np
import configparser
from utils import standard_transform, get_args


def get_adjacency_matrix(distance_df_filename: str, num_of_vertices: int, id_filename: str = None) -> tuple:
    """Generate adjacency matrix.

    Args:
        distance_df_filename (str): path of the csv file contains edges information
        num_of_vertices (int): number of vertices
        id_filename (str, optional): id filename. Defaults to None.

    Returns:
        tuple: two adjacency matrix.
            np.array: connectivity-based adjacency matrix A (A[i, j]=0 or A[i, j]=1)
            np.array: distance-based adjacency matrix A
    """

    if "npy" in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        adjacency_matrix_connectivity = np.zeros((int(num_of_vertices), int(
            num_of_vertices)), dtype=np.float32)
        adjacency_matrix_distance = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                                             dtype=np.float32)
        if id_filename:
            # the id in the distance file does not start from 0, so it needs to be remapped
            with open(id_filename, "r") as f:
                id_dict = {int(i): idx for idx, i in enumerate(
                    f.read().strip().split("\n"))}  # map node idx to 0-based index (start from 0)
            with open(distance_df_filename, "r") as f:
                f.readline()  # omit the first line
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[id_dict[i], id_dict[j]] = 1
                    adjacency_matrix_connectivity[id_dict[j], id_dict[i]] = 1
                    adjacency_matrix_distance[id_dict[i],
                                              id_dict[j]] = distance
                    adjacency_matrix_distance[id_dict[j],
                                              id_dict[i]] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance
        else:
            # ids in distance file start from 0
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[i, j] = 1
                    adjacency_matrix_connectivity[j, i] = 1
                    adjacency_matrix_distance[i, j] = distance
                    adjacency_matrix_distance[j, i] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance


def generate_adj(distance_df_filename, num_of_vertices, add_self_loop):
    if os.path.exists(distance_df_filename.split(".", maxsplit=1)[0] + ".txt"):
        id_filename = distance_df_filename.split(".", maxsplit=1)[0] + ".txt"
    else:
        id_filename = None
    adj_mx, distance_mx = get_adjacency_matrix(
        distance_df_filename, num_of_vertices, id_filename=id_filename)
    # the self loop is missing
    if add_self_loop:
        print("adding self loop to adjacency matrices.")
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])
    else:
        print("kindly note that there is no self loop in adjacency matrices.")
    
    return adj_mx, distance_mx
def process_dataset(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.
    Default settings of PEMS03 dataset:
        - Normalization method: standard norm.
        - Dataset division: 6:2:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic flow, time of day, day of week]
        - Target: predict the traffic speed of the future 12 time steps.

    Args:
        args (argparse): configurations of preprocessing
    """
    dataset_name = args.dataset_name
    mode = args.mode
    target_channel = [0]
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    mask_history_day = args.mask_history_day
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_self_loop = args.add_self_loop
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    distance_df_filename = args.distance_df_filename
    num_of_vertices = args.num_of_vertices
    steps_per_day = args.steps_per_day

    if mode == "PreTrain":
        # read data
        history_seq_len = mask_history_day * steps_per_day
        data = np.load(data_file_path)["data"]
        data = data[..., target_channel]
        print("raw time series shape: {0}".format(data.shape))

        l, n, f = data.shape
        num_samples = l - history_seq_len + 1
        train_num_short = round(num_samples * train_ratio)
        valid_num_short = round(num_samples * valid_ratio)
        test_num_short = num_samples - train_num_short - valid_num_short
        print("number of training samples:{0}".format(train_num_short))
        print("number of validation samples:{0}".format(valid_num_short))
        print("number of test samples:{0}".format(test_num_short))

        index_list = []
        for t in range(history_seq_len, num_samples + history_seq_len):
            index = (t-history_seq_len, t)
            index_list.append(index)

        train_index = index_list[:train_num_short]
        valid_index = index_list[train_num_short: train_num_short + valid_num_short]
        test_index = index_list[train_num_short + valid_num_short: train_num_short + valid_num_short + test_num_short]

        scaler = standard_transform(data, train_index)

        # label to identify the scaler for different settings.
        with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(scaler, f)

        # add external feature
        feature_list = [data]

        processed_data = np.concatenate(feature_list, axis=-1)  
        # dump data
        index = {}
        index["train"] = train_index
        index["valid"] = valid_index
        index["test"] = test_index
        with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(index, f)

        data = {}
        data["processed_data"] = processed_data
        with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(data, f)

        # generate and copy
        adj_mx, distance_mx = generate_adj(distance_df_filename, num_of_vertices, add_self_loop)
        with open(output_dir + "/adj_{0}.pkl".format(dataset_name), "wb") as f:
            pickle.dump(adj_mx, f)
        with open(output_dir + "/adj_{0}_distance.pkl".format(dataset_name), "wb") as f:
            pickle.dump(distance_mx, f)
    else:
        # read data
        data = np.load(data_file_path)["data"]
        data = data[..., target_channel]
        print("raw time series shape: {0}".format(data.shape))

        l, n, f = data.shape
        num_samples = l - (history_seq_len + future_seq_len) + 1
        train_num_short = round(num_samples * train_ratio)
        valid_num_short = round(num_samples * valid_ratio)
        test_num_short = num_samples - train_num_short - valid_num_short
        print("number of training samples:{0}".format(train_num_short))
        print("number of validation samples:{0}".format(valid_num_short))
        print("number of test samples:{0}".format(test_num_short))

        index_list = []
        for t in range(history_seq_len, num_samples + history_seq_len):
            index = (t-history_seq_len, t, t+future_seq_len)
            index_list.append(index)

        train_index = index_list[:train_num_short]
        valid_index = index_list[train_num_short: train_num_short + valid_num_short]
        test_index = index_list[train_num_short +
                                valid_num_short: train_num_short + valid_num_short + test_num_short]

        scaler = standard_transform(data, train_index)
        with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(scaler, f)
            
        # add external feature
        feature_list = [data]
        if add_time_of_day:
            # numerical time_of_day
            tod = [i % steps_per_day /
                steps_per_day for i in range(data.shape[0])]
            tod = np.array(tod)
            tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(tod_tiled)

        if add_day_of_week:
            # numerical day_of_week
            dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
            dow = np.array(dow)
            dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)

        processed_data = np.concatenate(feature_list, axis=-1)

        # dump data
        index = {}
        index["train"] = train_index
        index["valid"] = valid_index
        index["test"] = test_index
        with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(index, f)

        data = {}
        data["processed_data"] = processed_data
        with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(data, f)

        # generate and copy
        adj_mx, distance_mx = generate_adj(distance_df_filename, num_of_vertices, add_self_loop)
        with open(output_dir + "/adj_{0}.pkl".format(dataset_name), "wb") as f:
            pickle.dump(adj_mx, f)
        with open(output_dir + "/adj_{0}_distance.pkl".format(dataset_name), "wb") as f:
            pickle.dump(distance_mx, f)


if __name__ == "__main__":
    
    DATASET_NAME = "PeMS07"
    MODEL_NAME = "lsfgcn"
    MODE = "PreTrain"  # PreTrain or Train
    
    args = get_args(DATASET_NAME, MODEL_NAME, MODE)
    
    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/{0}/{0}.npz".format(DATASET_NAME)
    DISTANCE_DF_FILENAME = "datasets/{0}/{0}.csv".format(DATASET_NAME)
    
    args.output_dir = OUTPUT_DIR
    args.data_file_path = DATA_FILE_PATH
    args.distance_df_filename = DISTANCE_DF_FILENAME
    
    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    process_dataset(args)
