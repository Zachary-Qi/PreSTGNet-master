import os
import torch
import pickle
import random
import shutil
import logging
import argparse
import numpy as np
import configparser
from torch.utils.data import Dataset

def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)
    
def delete_target_folders(base_path):
    # 获取父文件夹和目标子文件夹路径
    parent_folder = os.path.dirname(os.path.dirname(base_path))  # 获取 'Train' 文件夹路径
    session_folder = os.path.dirname(base_path)  # 获取 '2025-02-21_10-28' 文件夹路径

    # 删除 session_1 文件夹
    if os.path.isdir(base_path):
        try:
            shutil.rmtree(base_path)  # 删除 'session_1' 文件夹及其内容
            print(f"Deleted folder: {base_path}")
        except Exception as e:
            print(f"Error deleting {base_path}: {e}")
    else:
        print(f"Path does not exist: {base_path}")

    # 删除 2025-02-21_10-28 文件夹
    if os.path.isdir(session_folder):
        try:
            shutil.rmtree(session_folder)  # 删除 '2025-02-21_10-28' 文件夹及其内容
            print(f"Deleted folder: {session_folder}")
        except Exception as e:
            print(f"Error deleting {session_folder}: {e}")
    else:
        print(f"Path does not exist: {session_folder}")

    # 确保 'Train' 文件夹还存在，不被删除
    if os.path.isdir(parent_folder):
        print(f"'Train' folder exists at: {parent_folder}")
    else:
        print(f"Path does not exist: {parent_folder}")
  
def aggregate_rank_step_metrics(local_rank, world_size, mae, mape, rmse, wmape, r2):
    
    # 在所有进程中聚合指标
    local_mae_tensor = torch.tensor(mae).to(torch.device("cuda", local_rank))
    local_mape_tensor = torch.tensor(mape).to(torch.device("cuda", local_rank))
    local_rmse_tensor = torch.tensor(rmse).to(torch.device("cuda", local_rank))
    local_wmape_tensor = torch.tensor(wmape).to(torch.device("cuda", local_rank))
    local_r2_tensor = torch.tensor(r2).to(torch.device("cuda", local_rank))
    # 使用all_reduce操作来聚合所有进程的mae
    torch.distributed.all_reduce(local_mae_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_mape_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_rmse_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_wmape_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_r2_tensor, op=torch.distributed.ReduceOp.SUM)
    
    # 获取平均值：所有进程的总和除以进程数量
    global_mae = local_mae_tensor / world_size
    global_mape = local_mape_tensor / world_size
    global_rmse = local_rmse_tensor / world_size
    global_wmape = local_wmape_tensor / world_size
    global_r2 = local_r2_tensor / world_size
    
    return global_mae.item(), global_mape.item(), global_rmse.item(), global_wmape.item(), global_r2.item()
  
        
def aggregate_rank_metrics(local_rank, world_size, local_metrics):
    # 通过all_reduce聚合指标值
    local_mae = np.mean(local_metrics["mae"])
    local_mape = np.mean(local_metrics["mape"])
    local_rmse = np.mean(local_metrics["rmse"])
    local_wmape = np.mean(local_metrics["wmape"])
    local_r2 = np.mean(local_metrics["r2"])
    # 在所有进程中聚合指标
    local_mae_tensor = torch.tensor(local_mae).to(torch.device("cuda", local_rank))
    local_mape_tensor = torch.tensor(local_mape).to(torch.device("cuda", local_rank))
    local_rmse_tensor = torch.tensor(local_rmse).to(torch.device("cuda", local_rank))
    local_wmape_tensor = torch.tensor(local_wmape).to(torch.device("cuda", local_rank))
    local_r2_tensor = torch.tensor(local_r2).to(torch.device("cuda", local_rank))
    # 使用all_reduce操作来聚合所有进程的mae
    torch.distributed.all_reduce(local_mae_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_mape_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_rmse_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_wmape_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_r2_tensor, op=torch.distributed.ReduceOp.SUM)
    
    # 获取平均值：所有进程的总和除以进程数量
    global_mae = local_mae_tensor / world_size
    global_mape = local_mape_tensor / world_size
    global_rmse = local_rmse_tensor / world_size
    global_wmape = local_wmape_tensor / world_size
    global_r2 = local_r2_tensor / world_size
    
    return global_mae.item(), global_mape.item(), global_rmse.item(), global_wmape.item(), global_r2.item()

def get_args(DATASET_NAME="PeMS08", MODEL_NAME = "lsfgcn", MODE="Train"):
    parser = argparse.ArgumentParser()
    
    # 设置配置文件路径
    parser.add_argument("--config", default='./configs/{}/{}.conf'.format(MODEL_NAME, DATASET_NAME), type=str,
                        help="configuration file path")
    
    args = parser.parse_args()
    
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # 配置数据相关参数
    data_config = config['Data']
    
    parser.add_argument("--num_of_vertices", type=int,
                        default=data_config['NUM_OF_VERTICES'], help="Number of vertices.")
    parser.add_argument("--history_seq_len", type=int,
                        default=data_config['HISTORY_SEQ_LEN'], help="History sequence length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=data_config['FUTURE_SEQ_LEN'], help="Future sequence length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=data_config['STEPS_PER_DAY'], help="Steps per day.")
    parser.add_argument("--mask_history_day", type=int,
                        default=data_config['MASK_HISTORY_DAY'], help="Steps per day.")
    
    parser.add_argument("--tod", type=bool,
                        default=data_config['TOD'], help="TOD.")
    parser.add_argument("--dow", type=bool,
                        default=data_config['DOW'], help="DOW.")
    parser.add_argument("--add_self_loop", type=bool,
                        default=data_config['ADD_SELF_LOOP'], help="ADD_SELF_LOOP.")
    
    parser.add_argument("--train_ratio", type=float,
                        default=data_config['TRAIN_RATIO'], help="TRAIN_RATIO.")
    parser.add_argument("--valid_ratio", type=float,
                        default=data_config['VALID_RATIO'], help="VALID_RATIO.")
    
    # 配置训练和预训练相关参数（通用）
    model_config_train = config['Training']
    model_config_pretrain = config['PreTraining']

    # Training config
    parser.add_argument("--runs_train", type=int,
                        default=model_config_train['RUNS_Train'], help="Runs for training.")
    parser.add_argument("--batch_size_train", type=int,
                        default=model_config_train['BATCH_SIZE_Train'], help="Batch size for training.")
    parser.add_argument("--in_channel_train", type=int,
                        default=model_config_train['IN_CHANNEL_Train'], help="Input channel for training.")
    parser.add_argument("--nb_chev_filter_train", type=int,
                        default=model_config_train['NB_CHEV_FILTER_Train'], help="Number of ChebNet filters for training.")
    parser.add_argument("--epochs_train", type=int,
                        default=model_config_train['EPOCHS_Train'], help="Number of epochs for training.")
    parser.add_argument("--start_epoch_train", type=int,
                        default=model_config_train['START_EPOCH_Train'], help="start_epoch_train.")
    parser.add_argument("--learning_rate_train", type=float,
                        default=model_config_train['LEARNING_RATE_Train'], help="Learning rate for training.")
    parser.add_argument("--dropout_train", type=float,
                        default=model_config_train['DROPOUT_Train'], help="Dropout rate for training.")
    parser.add_argument("--weight_decay_train", type=float,
                        default=model_config_train['WEIGHT_DECAY_Train'], help="Weight decay for training.")
    parser.add_argument("--alph", type=float,
                        default=model_config_train['ALPH_Train'], help="alph for training.")
    parser.add_argument("--gama", type=float,
                        default=model_config_train['GAMA_Train'], help="gama for training.")

    parser.add_argument("--es_patience_train", type=int,
                        default=model_config_train['ES_PATIENCE_Train'], help="Early stopping patience for training.")
    
    # PreTraining config
    parser.add_argument("--runs_pretrain", type=int,
                        default=model_config_pretrain['RUNS_Pre'], help="Runs for training.")
    parser.add_argument("--batch_size_pretrain", type=int,
                        default=model_config_pretrain['BATCH_SIZE_Pre'], help="Batch size for pretraining.")
    parser.add_argument("--in_channel_pretrain", type=int,
                        default=model_config_pretrain['IN_CHANNEL_Pre'], help="Input channel for pretraining.")
    parser.add_argument("--nb_chev_filter_pretrain", type=int,
                        default=model_config_pretrain['NB_CHEV_FILTER_Pre'], help="Number of ChebNet filters for pretraining.")
    parser.add_argument("--num_heads_pretrain", type=int,
                        default=model_config_pretrain['NUM_HEADS_Pre'], help="Number of attention heads for pretraining.")
    parser.add_argument("--mlp_ratio_pretrain", type=int,
                        default=model_config_pretrain['MLP_RATIO_Pre'], help="MLP ratio for pretraining.")
    parser.add_argument("--epochs_pretrain", type=int,
                        default=model_config_pretrain['EPOCHS_Pre'], help="Number of epochs for pretraining.")
    parser.add_argument("--start_epoch_pretrain", type=int,
                        default=model_config_pretrain['START_EPOCH_Pre'], help="start_epoch_pretrain.")
    parser.add_argument("--mask_ratio_pretrain", type=float,
                        default=model_config_pretrain['MASK_RATIO_Pre'], help="Mask ratio for pretraining.")
    parser.add_argument("--dropout_pretrain", type=float,
                        default=model_config_pretrain['DROPOUT_Pre'], help="Dropout rate for pretraining.")
    parser.add_argument("--learning_rate_pretrain", type=float,
                        default=model_config_pretrain['LEARNING_RATE_Pre'], help="Learning rate for pretraining.")
    parser.add_argument("--weight_decay_pretrain", type=float,
                        default=model_config_pretrain['WEIGHT_DECAY_Pre'], help="Weight decay for pretraining.")
    parser.add_argument("--num_of_hours_pretrain", type=int,
                        default=model_config_pretrain['NUM_OF_HOURS_Pre'], help="Number of hours for pretraining.")
    parser.add_argument("--es_patience_pretrain", type=int,
                        default=model_config_pretrain['ES_PATIENCE_Pre'], help="Early stopping patience for pretraining.")
    parser.add_argument("--encoder_depth_pretrain", type=int,
                        default=model_config_pretrain['ENCODER_DEPTH_Pre'], help="Encoder depth for pretraining.")
    parser.add_argument("--decoder_depth_pretrain", type=int,
                        default=model_config_pretrain['DECODER_DEPTH_Pre'], help="Decoder depth for pretraining.")
    
    args = parser.parse_args()
    
    # 设置模式和数据集名称
    args.mode = MODE
    args.dataset_name = DATASET_NAME
    
    return args

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    
def standard_transform(data: np.array, train_index: list):
    """Standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        history_seq_len (int): historical sequence length.
        future_seq_len (int): future sequence length.

    Returns:
        np.array: normalized raw time series data.
    """

    # data: L, N, C, C=1
    data_train = data[:train_index[-1][1], ...]

    mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)
    scaler = {}
    scaler["func"] = re_standard_transform.__name__
    scaler["args"] = {"mean": mean, "std": std}
    
    return scaler
   

def normalize(x, mean, std):
    return (x - mean) / std


def re_standard_transform(data: torch.Tensor, scaler) -> torch.Tensor:
    """Standard re-transformation.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """
    mean, std = scaler["mean"], scaler["std"]
    data = data * std
    data = data + mean
    return data

class prepare_forecasting_data(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, seq_len:int, scaler, train_mode: str) -> None:
        """Init the dataset in the forecasting stage.

        Args:
            data_file_path (str): data file path.
            index_file_path (str): index file path.
            mode (str): train, valid, or test.
            seq_len (int): the length of long term historical data.
        """

        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        self.train_mode = train_mode
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        self.scaler = scaler
        # read index
        self.index = load_pkl(index_file_path)[mode]
        # length of long term historical data
        self.seq_len = seq_len
        # mask
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], 1)

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """
        
        idx = list(self.index[index])
        
        if self.train_mode == "PreTrain":
            # 对训练数据进行标准化处理
            history_data = self.data[idx[0]:idx[1]]     # 12
            history_data_norm = normalize(history_data, self.scaler["mean"], self.scaler["std"])
            return history_data_norm
        else:
            history_data = self.data[idx[0]:idx[1]]     # 12
            history_data_norm_0 = normalize(history_data[:,:,[0]], self.scaler["mean"], self.scaler["std"])
            future_data = self.data[idx[1]:idx[2]][:,:,[0]]     # 12
            if idx[1] - self.seq_len < 0:
                long_history_data = self.mask
            else:
                long_history_data = self.data[idx[1] - self.seq_len:idx[1]][:,:,[0]]     # 11
            long_history_data_norm = normalize(long_history_data, self.scaler["mean"], self.scaler["std"])
            
            history_data_norm = torch.cat([history_data_norm_0, history_data[:,:,[1]], history_data[:,:,[2]]], dim=2)
            return history_data_norm, future_data, long_history_data_norm
        
    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """
        return len(self.index)
    
def delete_files_with_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted {file_path}")

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)
    # TODO fix very large values
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean squared error
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """root mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.

    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss

def R2_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    ss_res = torch.sum((true - pred) ** 2)
    ss_tot = torch.sum((true - torch.mean(true)) ** 2)
    return 1 - ss_res / ss_tot

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    r2 = R2_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape,r2

def setup_logging(log_dir='logs', log_file='data_processing.log'):
    """
    配置日志记录器
    :param log_dir: 日志文件夹路径
    :param log_file: 日志文件名
    :return: 配置好的日志记录器
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def load_pkl(pickle_file: str) -> object:
    """Load pickle data.
    Args:
        pickle_file (str): file path
    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


