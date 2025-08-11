import os
import gc
import time
import copy
import torch
import torch.distributed as dist

from tqdm import tqdm
from datetime import datetime
from lib.engine1 import trainer
from torch.utils.data import DataLoader, DistributedSampler
from lib.utils import *


# 加载配置文件
DATASET_NAME = "PeMS04"
MODEL_NAME = "lsfgcn"
MODE = "Train" # only Train or PreTrain
# 加载训练好的掩码模型
PRE_TRAINED_WEIGHT_PATH = "./experiments/PeMS04_lsfgcn/PreTrain/2025-04-28_17-50/session_1/checkpoint.pth"
# 断点训练-断点路径
Breakpoint_Destination_Folder = "./experiments/PeMS03_lsfgcn/PreTrain/2025-04-16_12-59/session_1/"

def trainer_distributed(args, local_rank, world_size, train_loader, valid_loader, test_loader, scaler, logger):
    # Initialise seed
    seed_it(6666)
    engine = trainer(args, scaler, local_rank, logger)

    global_min_loss = 999999.
    global_update_count = 0
    
    if args.mode == "PreTrain":
        start_epoch = args.start_epoch_pretrain
        epochs = args.epochs_pretrain
        es_patience = args.es_patience_pretrain
    else:
        start_epoch = args.start_epoch_train
        epochs = args.epochs_train
        es_patience = args.es_patience_train
        
    if 0 < start_epoch < epochs:
        if local_rank == 0:
            delete_target_folders(args.save_dir)
            
        args.save_dir = Breakpoint_Destination_Folder
        
        out_dir = os.path.dirname(Breakpoint_Destination_Folder)
        log_file = "{}_breakpoint.log".format(MODE)
        logger = setup_logging(out_dir +"/logs", log_file)
        
        if args.mode == "PreTrain":
            checkpoint = torch.load(os.path.join(args.save_dir, "checkpoint.pth"), weights_only=True)

            engine.model.load_state_dict(checkpoint['model_state_dict'])
            engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_min_loss = checkpoint['loss']
            start_epoch = checkpoint['epoch']
            
        else:
            checkpoint = torch.load(os.path.join(args.save_dir, "checkpoint.pth"), weights_only=True)
            
            engine.predict_model.load_state_dict(checkpoint['model_state_dict'])
            engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_min_loss = checkpoint['loss']
            start_epoch = checkpoint['epoch']
    
    # for epoch in range(start_epoch, epochs):
        
    #     train_start_time = time.time()
    #     train_metrics = {
    #         "mae": [],
    #         "mape": [],
    #         "rmse": [],
    #         "wmape": [],
    #         "r2": []
    #     }
        
    #     with tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress", unit="step", disable=(local_rank != 0), leave=False) as pbar:
    #         for iter, dataCollections in pbar:
    #             metrics = engine.train(dataCollections)
                
    #             train_metrics["mae"].append(metrics[0])
    #             train_metrics["mape"].append(metrics[1])
    #             train_metrics["rmse"].append(metrics[2])
    #             train_metrics["wmape"].append(metrics[3])
    #             train_metrics["r2"].append(metrics[4])
                
    #             pbar.set_postfix({
    #                 "mae": f"{metrics[0]:.2f}",
    #                 "rmse": f"{metrics[2]:.2f}",
    #                 "mape": f"{metrics[1]*100:.2f}%",
    #                 "r2": f"{metrics[4]:.2f}"
    #             })
    #     train_end_time = time.time()
        
    #     global_train_metrics = aggregate_rank_metrics(local_rank, world_size, train_metrics)
        
    #     # 计算训练所用时间
    #     training_time = train_end_time - train_start_time
        
    #     # 打印平均值和训练时间
    #     if local_rank == 0:
    #         logger.info("Epoch: {:03d}, Average Metrics: Training MAE: {:.4f}, Training MAPE: {:.4f}, Training RMSE: {:.4f}, Training WMAPE: {:.4f}, Training r2: {:.4f}, Training Time: {:.4f} secs".format(
    #                 epoch, global_train_metrics[0], global_train_metrics[1], global_train_metrics[2], global_train_metrics[3], global_train_metrics[4], training_time))
            
    #     valid_start_time = time.time()
    #     valid_metrics = {
    #         "mae": [],
    #         "mape": [],
    #         "rmse": [],
    #         "wmape": [],
    #         "r2": []
    #     }
        
    #     with tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Valid Progress", unit="step", disable=(local_rank != 0), leave=False) as pbar:
    #         for iter, dataCollections in pbar:
    #             metrics = engine.eval(dataCollections)
                
    #             valid_metrics["mae"].append(metrics[0])
    #             valid_metrics["mape"].append(metrics[1])
    #             valid_metrics["rmse"].append(metrics[2])
    #             valid_metrics["wmape"].append(metrics[3])
    #             valid_metrics["r2"].append(metrics[4])
                
    #             pbar.set_postfix({
    #                 "mae": f"{metrics[0]:.2f}",
    #                 "rmse": f"{metrics[2]:.2f}",
    #                 "mape": f"{metrics[1]*100:.2f}%",
    #                 "r2": f"{metrics[4]:.2f}"
    #             })
                    
    #     valid_end_time = time.time()
        
    #     # 计算每个指标的平均值
    #     global_valid_metrics = aggregate_rank_metrics(local_rank, world_size, valid_metrics)
        
    #     # 计算训练所用时间
    #     validing_time = valid_end_time - valid_start_time
        
    #     if global_valid_metrics[0] < global_min_loss:
            
    #         # 同步进程
    #         dist.barrier()
    #         # 保存模型
    #         save_model = engine.predict_model if args.mode == "Train" else engine.model
            
    #         if local_rank == 0:
    #             new_loss = global_valid_metrics[0]
    #             logger.info(f"Update tasks appear: {global_min_loss:.4f} > {new_loss:.4f}, -{global_min_loss-new_loss:.4f}")
    #             logger.info("Epoch: {:03d}, Average Metrics: validing MAE: {:.4f}, validing MAPE: {:.4f}, validing RMSE: {:.4f}, validing WMAPE: {:.4f}, validing r2: {:.4f}, validing Time: {:.4f} secs".format(
    #                 epoch, global_valid_metrics[0], global_valid_metrics[1], global_valid_metrics[2], global_valid_metrics[3], global_valid_metrics[4], validing_time))

    #             global_min_loss = global_valid_metrics[0]
    #             global_update_count = 0
                
    #             best_model = copy.deepcopy(save_model.state_dict())
    #             best_optimizer = copy.deepcopy(engine.optimizer.state_dict())
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': best_model,
    #                 'optimizer_state_dict': best_optimizer,
    #                 'loss': global_min_loss,
    #             }, os.path.join(args.save_dir, "checkpoint.pth"))
            
    #     else:
    #         # 打印平均值和训练时间
    #         # 同步进程
    #         dist.barrier()
    #         if local_rank == 0:
    #             global_update_count += 1
    #             logger.debug(f"Cumulative total of {global_update_count} No update tasks appear")
    #             logger.info("Epoch: {:03d}, Average Metrics: validing MAE: {:.4f}, validing MAPE: {:.4f}, validing RMSE: {:.4f}, validing WMAPE: {:.4f}, validing r2: {:.4f}, validing Time: {:.4f} secs".format(
    #                 epoch, global_valid_metrics[0], global_valid_metrics[1], global_valid_metrics[2], global_valid_metrics[3], global_valid_metrics[4], validing_time))
    
    # dist.barrier()  # 等待所有进程同步到这里
    # if local_rank == 0:
    #     logger.info("Training finished.")

    # 加载模型，优化器，和Scaler的状态
    save_dir = "./experiments/PeMS04_lsfgcn/Train/2025-04-29_14-33/session_1"
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth"), weights_only=True)
    
    # checkpoint = torch.load(os.path.join(args.save_dir, "checkpoint.pth"), weights_only=True)

    save_model = engine.predict_model if args.mode == "Train" else engine.model
    
    save_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_start_time = time.time()
    
    test_metrics = {
            "mae": [],
            "mape": [],
            "rmse": [],
            "wmape": [],
            "r2": []
        }
    
    predicts, reals = [], []
    with tqdm(enumerate(test_loader), total=len(test_loader), desc="Test Progress", unit="step", disable=(local_rank != 0), leave=False) as pbar:
        for iter, dataCollections in pbar:
            predict, real = engine.test(dataCollections)
            predicts.append(predict)
            reals.append(real)
    
    predicts_all = torch.cat(predicts, dim=0)
    reals_all = torch.cat(reals, dim=0)
    
    if args.mode == "PreTrain": 
        mae = masked_mae(predicts_all, reals_all).item()
        mape = masked_mape(predicts_all, reals_all, 0.0).item()
        rmse = masked_rmse(predicts_all, reals_all, 0.0).item()
        wmape = WMAPE_torch(predicts_all, reals_all, 0.0).item()
        r2 = R2_torch(predicts_all, reals_all, 0.0).item()
        
        test_metrics["mae"].append(mae)
        test_metrics["mape"].append(mape)
        test_metrics["rmse"].append(rmse)
        test_metrics["wmape"].append(wmape)
        test_metrics["r2"].append(r2)

        # 同步进程
        dist.barrier()
        mae, mape, rmse, wmape, r2 = aggregate_rank_step_metrics(local_rank, world_size, mae, mape, rmse, wmape, r2)
        
        if local_rank == 0:
            logger.info(
                f"Average for Test: Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}, Test WMAPE: {wmape:.4f}, Test R2: {r2:.4f}")
    else:
        for t in range(reals_all.shape[1]):
            
            mae = masked_mae(predicts_all[:,t,:,:], reals_all[:,t,:,:]).item()
            mape = masked_mape(predicts_all[:,t,:,:], reals_all[:,t,:,:], 0.0).item()
            rmse = masked_rmse(predicts_all[:,t,:,:], reals_all[:,t,:,:], 0.0).item()
            wmape = WMAPE_torch(predicts_all[:,t,:,:], reals_all[:,t,:,:], 0.0).item()
            r2 = R2_torch(predicts_all[:,t,:,:], reals_all[:,t,:,:], 0.0).item()
            
            test_metrics["mae"].append(mae)
            test_metrics["mape"].append(mape)
            test_metrics["rmse"].append(rmse)
            test_metrics["wmape"].append(wmape)
            test_metrics["r2"].append(r2)

            # 同步进程
            dist.barrier()
            mae, mape, rmse, wmape, r2 = aggregate_rank_step_metrics(local_rank, world_size, mae, mape, rmse, wmape, r2)
            
            if local_rank == 0:
                logger.info(
                    f"Average for Horizon {t+1}: Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}, Test WMAPE: {wmape:.4f}, Test R2: {r2:.4f}")
        
        # 同步进程
        dist.barrier()
        global_test_metrics = aggregate_rank_metrics(local_rank, world_size, test_metrics)
        
        test_end_time = time.time()
        
        # 计算训练所用时间
        test_time = test_end_time - test_start_time
        
        if local_rank == 0:
            logger.info("Average Metrics: test MAE: {:.4f}, test MAPE: {:.4f}, test RMSE: {:.4f}, test WMAPE: {:.4f}, test r2: {:.4f}, test Time: {:.4f} secs".format(
                global_test_metrics[0], global_test_metrics[1], global_test_metrics[2], global_test_metrics[3], global_test_metrics[4], test_time))
    

def main(args, logger, local_rank, world_size):
    
    OUTPUT_DIR = "datasets/" + args.dataset_name

    batch_size = 0
    # 判别训练模式，加载对应的数据集
    if args.mode == "PreTrain":
        
        batch_size = args.batch_size_pretrain
        
        data_file_path = OUTPUT_DIR+"/data_in{0}_out{1}.pkl".format(args.mask_history_day * args.steps_per_day, args.future_seq_len)
        index_file_path = OUTPUT_DIR+"/index_in{0}_out{1}.pkl".format(args.mask_history_day * args.steps_per_day, args.future_seq_len)
        scaler_file_path = OUTPUT_DIR+"/scaler_in{0}_out{1}.pkl".format(args.mask_history_day * args.steps_per_day, args.future_seq_len)

        scaler = load_pkl(scaler_file_path)["args"]
        
        train_dataset = prepare_forecasting_data(data_file_path, index_file_path, "train", args.mask_history_day * args.steps_per_day, scaler, train_mode=args.mode)
        valid_dataset = prepare_forecasting_data(data_file_path, index_file_path, "valid", args.mask_history_day * args.steps_per_day, scaler, train_mode=args.mode)
        test_dataset = prepare_forecasting_data(data_file_path, index_file_path, "test", args.mask_history_day * args.steps_per_day, scaler, train_mode=args.mode)
        
        if local_rank == 0:
            # 打印 train_dataset, valid_dataset, test_dataset 的 shape
            logger.info(f"train_X shape: [{len(train_dataset), args.mask_history_day * args.steps_per_day, train_dataset[0][0].shape[0], train_dataset[0][0].shape[1]}]")
            logger.info(f"valid_X shape: [{len(valid_dataset), args.mask_history_day * args.steps_per_day, valid_dataset[0][0].shape[0], valid_dataset[0][0].shape[1]}]")
            logger.info(f"test_X shape: [{len(test_dataset), args.mask_history_day * args.steps_per_day, test_dataset[0][0].shape[0], test_dataset[0][0].shape[1]}]")

    else:
        
        batch_size = args.batch_size_train
        
        data_file_path = OUTPUT_DIR+"/data_in{0}_out{1}.pkl".format(args.history_seq_len, args.future_seq_len)
        index_file_path = OUTPUT_DIR+"/index_in{0}_out{1}.pkl".format(args.history_seq_len, args.future_seq_len)
        scaler_file_path = OUTPUT_DIR+"/scaler_in{0}_out{1}.pkl".format(args.history_seq_len, args.future_seq_len)

        scaler = load_pkl(scaler_file_path)["args"]
        
        train_dataset = prepare_forecasting_data(data_file_path, index_file_path, "train", args.mask_history_day * args.steps_per_day, scaler, train_mode=args.mode)
        valid_dataset = prepare_forecasting_data(data_file_path, index_file_path, "valid", args.mask_history_day * args.steps_per_day, scaler, train_mode=args.mode)
        test_dataset = prepare_forecasting_data(data_file_path, index_file_path, "test", args.mask_history_day * args.steps_per_day, scaler, train_mode=args.mode)

        if local_rank == 0:
            # 打印 train_dataset, valid_dataset, test_dataset 的 shape
            logger.info(f"train_X shape: [{len(train_dataset), train_dataset[0][0].shape[0], train_dataset[0][0].shape[1], train_dataset[0][0].shape[2]}], train_y shape: [{len(train_dataset), train_dataset[1][1].shape[0], train_dataset[1][1].shape[1], train_dataset[1][1].shape[2]}], train_long shape: [{len(train_dataset), train_dataset[2][2].shape[0], train_dataset[2][2].shape[1], train_dataset[2][2].shape[2]}]")
            logger.info(f"valid_X shape: [{len(valid_dataset), valid_dataset[0][0].shape[0], valid_dataset[0][0].shape[1], valid_dataset[0][0].shape[2]}], valid_y shape: [{len(valid_dataset), valid_dataset[1][1].shape[0], valid_dataset[1][1].shape[1], valid_dataset[1][1].shape[2]}], valid_long shape: [{len(valid_dataset), valid_dataset[2][2].shape[0], valid_dataset[2][2].shape[1], valid_dataset[2][2].shape[2]}]")
            logger.info(f"test_X shape: [{len(test_dataset), test_dataset[0][0].shape[0], test_dataset[0][0].shape[1], test_dataset[0][0].shape[2]}], test_y shape: [{len(test_dataset), test_dataset[1][1].shape[0], test_dataset[1][1].shape[1], test_dataset[1][1].shape[2]}], test_long shape: [{len(test_dataset), test_dataset[2][2].shape[0], test_dataset[2][2].shape[1], test_dataset[2][2].shape[2]}]")

    # 使用 DistributedSampler 确保每个进程都加载不同的数据子集
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)
    
    del train_dataset, valid_dataset, test_dataset
    gc.collect()  # 手动触发垃圾回收
    
    trainer_distributed(args, local_rank, world_size, train_loader, valid_loader, test_loader, scaler, logger)
    
if __name__ == "__main__":
    
    # 获取环境变量 LOCAL_RANK 和 WORLD_SIZE
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])  # 进程总数
    
    args = get_args(DATASET_NAME, MODEL_NAME, MODE)

    args.pre_trained_weight_path = PRE_TRAINED_WEIGHT_PATH
    CURRENT_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_dir = "./experiments/{0}_{1}/{2}/{3}".format(DATASET_NAME, MODEL_NAME, MODE, CURRENT_TIME)
    log_file = "{}.log".format(MODE)
    logger = setup_logging(out_dir +"/logs", log_file)
    logger.debug("{} is being trained using {}.".format(DATASET_NAME, MODEL_NAME))
    
    logger.debug("Training on {} local_rank".format(local_rank))
    logger.debug("Training on {} GPUs".format(world_size))
    
    if local_rank == 0:
        logger.info("-" * (20 + 45 + 5))
        for key, value in sorted(vars(args).items()):
            logger.info("|{0:>20} = {1:<45}|".format(key, str(value)))
        logger.info("-" * (20 + 45 + 5))
    
    runs = 0
    
    if args.mode == "Train":
        runs = args.runs_train
    else:
        runs = args.runs_pretrain
    
    t1 = time.time()
    for i in range(runs):
        if local_rank == 0:
            logger.info("The {} training session is initiated....".format(i+1))
            
        args.save_dir = os.path.join(out_dir, "session_" + str(i+1))
        os.makedirs(args.save_dir, exist_ok=True)
        
        metric_step = main(args, logger, local_rank, world_size)
        if local_rank == 0:
            logger.info("The {} training session is complete!".format(i+1))

    t2 = time.time()
    if local_rank == 0:
        logger.info("Total time spent: {:.4f}".format(t2 - t1))