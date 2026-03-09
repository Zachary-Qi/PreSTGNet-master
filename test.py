import os
import gc
import time
import copy
import torch
import torch.distributed as dist

from tqdm import tqdm
from datetime import datetime
from lib.engine import trainer
from torch.utils.data import DataLoader, DistributedSampler
from lib.utils import *


DATASET_NAME = "PeMS08"
MODEL_NAME = "PreSTGNet"
MODE = "Train" # only Train or PreTrain

PRE_TRAINED_WEIGHT_PATH = "./experiments/PeMS08_PreSTGNet/PreTrain/2026-02-08_03-47/session_1/checkpoint.pth"

save_dir = "./experiments/PeMS08_PreSTGNet/Train/2026-02-10_02-08/session_1/" 

def trainer_distributed(args, local_rank, world_size, train_loader, valid_loader, test_loader, scaler, logger):
    # Initialise seed
    seed_it(6666)
    engine = trainer(args, scaler, local_rank, logger)
    
    if args.mode == "PreTrain":
        checkpoint = torch.load(os.path.join(args.save_dir, "checkpoint.pth"), weights_only=True)

        engine.model.load_state_dict(checkpoint['model_state_dict'])
        engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    else:
        checkpoint = torch.load(os.path.join(args.save_dir, "checkpoint.pth"), weights_only=True)
        
        engine.predict_model.load_state_dict(checkpoint['model_state_dict'])
        engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    dist.barrier() 
    if local_rank == 0:
        logger.info("Training finished.")
        
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth"), weights_only=True)
    
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

            dist.barrier()
            mae, mape, rmse, wmape, r2 = aggregate_rank_step_metrics(local_rank, world_size, mae, mape, rmse, wmape, r2)
            
            if local_rank == 0:
                logger.info(
                    f"Average for Horizon {t+1}: Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}, Test WMAPE: {wmape:.4f}, Test R2: {r2:.4f}")
        
        dist.barrier()
        global_test_metrics = aggregate_rank_metrics(local_rank, world_size, test_metrics)
        
        test_end_time = time.time()
        
        test_time = test_end_time - test_start_time
        
        if local_rank == 0:
            logger.info("Average Metrics: test MAE: {:.4f}, test MAPE: {:.4f}, test RMSE: {:.4f}, test WMAPE: {:.4f}, test r2: {:.4f}, test Time: {:.4f} secs".format(
                global_test_metrics[0], global_test_metrics[1], global_test_metrics[2], global_test_metrics[3], global_test_metrics[4], test_time))
    

def main(args, logger, local_rank, world_size):
    
    OUTPUT_DIR = "datasets/" + args.dataset_name

    batch_size = 0
 
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

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)
    
    del train_dataset, valid_dataset, test_dataset
    gc.collect()  
    
    trainer_distributed(args, local_rank, world_size, train_loader, valid_loader, test_loader, scaler, logger)
    
if __name__ == "__main__":
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])  
    
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
            
        args.save_dir = save_dir
        os.makedirs(args.save_dir, exist_ok=True)
        
        metric_step = main(args, logger, local_rank, world_size)
        if local_rank == 0:
            logger.info("The {} training session is complete!".format(i+1))

    t2 = time.time()
    if local_rank == 0:
        logger.info("Total time spent: {:.4f}".format(t2 - t1))
