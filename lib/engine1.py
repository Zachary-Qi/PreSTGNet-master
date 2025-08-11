import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta  # 添加这一行来导入 timedelta

from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils import *
from ranger21 import Ranger
from modules.lsfgcn_best.mask_pretrain import Mask  
from modules.lsfgcn_best.lsst_fusion import DSTIGFN 


def setup_distributed_backend_run(args, local_rank):
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=1800))  # 适当增加超时时间
    torch.cuda.set_device(local_rank)  # 每个进程使用一个 GPU
    
    if args.mode == "PreTrain":
        model = Mask(
                patch_size=args.future_seq_len,
                in_channel=args.in_channel_pretrain,
                embed_dim=args.nb_chev_filter_pretrain,
                num_nodes=args.num_of_vertices,
                num_heads=args.num_heads_pretrain,
                mlp_ratio=args.mlp_ratio_pretrain,
                dropout=args.dropout_pretrain,
                mask_ratio=args.mask_ratio_pretrain,
                encoder_depth=args.encoder_depth_pretrain,
                decoder_depth=args.decoder_depth_pretrain,
                mode="pre-train"
            ).cuda()
        
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        return model, None
    else:
        model = Mask(
                    patch_size=args.future_seq_len,
                    in_channel=args.in_channel_pretrain,
                    embed_dim=args.nb_chev_filter_pretrain,
                    num_nodes=args.num_of_vertices,
                    num_heads=args.num_heads_pretrain,
                    mlp_ratio=args.mlp_ratio_pretrain,
                    dropout=args.dropout_pretrain,
                    mask_ratio=args.mask_ratio_pretrain,
                    encoder_depth=args.encoder_depth_pretrain,
                    decoder_depth=args.decoder_depth_pretrain,
                    mode="forecasting"
                ).cuda()
        
        predict_model = DSTIGFN(args.history_seq_len, args.future_seq_len, args.in_channel_train, args.num_of_vertices, 
                                args.nb_chev_filter_train, args.steps_per_day, args.steps_per_day * args.mask_history_day, args.dropout_train, args.alph, args.gama).cuda()

        # 使用 DistributedDataParallel
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        predict_model = DDP(predict_model, device_ids=[local_rank], find_unused_parameters=False)
        
        return model, predict_model


class trainer:
    def __init__(self, args, scaler, local_rank, logger):
        
        self.model, self.predict_model = setup_distributed_backend_run(args, local_rank)
        self.loss = nn.L1Loss()  # 使用 MAE 损失函数
        self.scaler = scaler
        self.clip = 5
        self.mode = args.mode
        if self.mode == "Train":
            self.optimizer = Ranger(self.predict_model.parameters(), lr=args.learning_rate_train, weight_decay=args.weight_decay_train)
            self.pre_trained_weight_path = args.pre_trained_weight_path
            self.load_pre_trained_model()
            logger.info(self.predict_model)
        else:
            self.optimizer = Ranger(self.model.parameters(), lr=args.learning_rate_pretrain, weight_decay=args.weight_decay_pretrain)
            logger.info(self.model)

    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint = torch.load(os.path.join(self.pre_trained_weight_path), weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def train(self, dataCollections):
        
        if self.mode == "Train":
        
            input, real, input_z = dataCollections[0].cuda(), dataCollections[1].cuda(), dataCollections[2].cuda()
            # print("input:{}, real:{}, input_z:{}".format(input.shape, real.shape, input_z.shape))
            self.predict_model.train()
            self.optimizer.zero_grad()
            hidden_states_full = self.model(input_z)
            # 进入训练模型
            output = self.predict_model(input, hidden_states_full)
            predict = re_standard_transform(output, self.scaler)
            loss = self.loss(predict, real)

            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.predict_model.parameters(), self.clip)
        else:
            input = dataCollections.cuda()
            
            self.model.train()
            self.optimizer.zero_grad()
            # 进入训练模型
            output_x, output_y = self.model(input)
            
            predict = re_standard_transform(output_x, self.scaler)
            real = re_standard_transform(output_y, self.scaler)
            
            loss = self.loss(predict, real)

            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        
        self.optimizer.step()
        
        mae = loss.item()
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        wmape = WMAPE_torch(predict, real, 0.0).item()
        r2 = R2_torch(predict, real, 0.0).item()
        
        return mae, mape, rmse, wmape, r2

    def eval(self, dataCollections):
        
        if self.mode == "Train":
        
            input, real, input_z = dataCollections[0].cuda(), dataCollections[1].cuda(), dataCollections[2].cuda()
            self.predict_model.eval()
            with torch.no_grad():
                hidden_states_full = self.model(input_z)
                # 进入训练模型
                output = self.predict_model(input, hidden_states_full)
                predict = re_standard_transform(output, self.scaler)
                loss = self.loss(predict, real)
        else:
            input = dataCollections.cuda()
            
            self.model.eval()
            with torch.no_grad():
                # 进入训练模型
                output_x, output_y = self.model(input)
                
                predict = re_standard_transform(output_x, self.scaler)
                real = re_standard_transform(output_y, self.scaler)
            
                loss = self.loss(predict, real)

        mae = loss.item()
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        wmape = WMAPE_torch(predict, real, 0.0).item()
        r2 = R2_torch(predict, real, 0.0).item()
        
        return mae, mape, rmse, wmape, r2

    def test(self, dataCollections):
        if self.mode == "Train":
        
            input, real, input_z = dataCollections[0].cuda(), dataCollections[1].cuda(), dataCollections[2].cuda()
            self.predict_model.eval()
            with torch.no_grad():
                hidden_states_full = self.model(input_z)
                # 进入训练模型
                output = self.predict_model(input, hidden_states_full)
                predict = re_standard_transform(output, self.scaler)
        else:
            input = dataCollections.cuda()
            
            self.model.eval()
            with torch.no_grad():
                # 进入训练模型
                output_x, output_y = self.model(input)
                
                predict = re_standard_transform(output_x, self.scaler)
                real = re_standard_transform(output_y, self.scaler)

        return predict, real