import os
import gc
import time

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import pandas as pd
import numpy as np
from tqdm import tqdm

from monai.inferers import sliding_window_inference

from src.data.utils import get_dataset, get_dataloader
from src.models.utils import get_model, ModelEMA

from src.modules.utils import (
    get_optimizer,
    get_scheduler,
    batch_to_device,
    calc_grad_norm,
    flatten_dict,
    save_weights,
)

from src.logging.utils import get_logger
from src.utils.torch import nms_3d
from src.modules.metric import score

def run_eval(model, val_ds, val_dl, val_metrics, cfg):
    if cfg.world_size > 1:
        dist.barrier()
    model.eval()
    
    progress_bar = tqdm(range(len(val_dl)), disable=cfg.local_rank!=0)
    val_itr = iter(val_dl)
    val_acc= 0
    i= 0
    logits= []
    targets= []
    losses= []
    max_preds= []

    with torch.no_grad():
        for itr in progress_bar:
            with autocast(cfg.device.type):

                batch= next(val_itr)
                if cfg.world_size > 1:
                    batch = batch_to_device(batch, cfg.local_rank) 
                else:
                    batch = batch_to_device(batch, cfg.device) 

                # Sliding window
                batch["input"]= batch["input"].float()
                preds = sliding_window_inference(
                    inputs= batch["input"],
                    roi_size= cfg.roi_size,
                    predictor= model,
                    overlap= 0.25,
                    sw_batch_size= 1,
                )

                # Loss
                if cfg.world_size > 1:
                    loss_fn = model.module.loss_fn
                else:
                    loss_fn= model.loss_fn

                loss = loss_fn(
                    input=preds,
                    target=batch["target"].float(),
                ).item()
                losses.append(loss)
                preds= preds[0, 0, ...]

                # Get argmax prediction
                amax_idx = torch.argmax(preds)
                coords = torch.unravel_index(amax_idx, preds.shape)
                prob = torch.sigmoid(preds[coords])
                max_preds.append({
                    "z": coords[0].item(),
                    "y": coords[1].item(),
                    "x": coords[2].item(),
                    "prob": prob.item(),
                    })


    if len(max_preds) == 0:
        return val_metrics

    # Loss
    val_metrics["val"]["loss"]= np.mean(losses)

    # Competition metric
    thresh= 0.5
    col_map= {
        "z": "Motor axis 0",
        "y": "Motor axis 1",
        "x": "Motor axis 2",
        "n_motors": "Has motor",
        "voxel_spacing": "Voxel spacing"
    }

    # Solution
    if isinstance(val_ds, torch.utils.data.Subset):
        sol= val_ds.dataset.df.copy()
    else:
        sol= val_ds.df.copy()
    sol= sol.rename(columns=col_map)
    
    # Submission
    sub= pd.DataFrame(max_preds)
    sub["tomo_id"]= sol["tomo_id"]

    sub[["z", "y", "x"]]= sub[["z", "y", "x"]] / preds.shape
    sub["z"] = sub["z"].values * sol["z_shape"].values
    sub["y"] = sub["y"].values * sol["y_shape"].values
    sub["x"] = sub["x"].values * sol["x_shape"].values

    # Threshold
    cutoff_pct= np.mean(sol["Has motor"].mean()) * 0.75
    cutoff= sub['prob'].quantile(cutoff_pct)
    sub.loc[sub["prob"] <= cutoff, ["z", "y", "x"]]= -1.0
    print("="*25)
    print("cutoff:", cutoff)
    print("cutoff_pct:", cutoff_pct)
    print("="*25)

    # Score
    sol= sol.rename(columns=col_map)
    sub= sub.rename(columns=col_map)
    sol= sol[["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2", "Voxel spacing", "Has motor"]]
    sub= sub[["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]]

    comp_metrics= score(sol, sub, min_radius=1000, beta=2)
    val_metrics["val"] |= comp_metrics

    # NOTE: DDP not supported
    if cfg.world_size > 1:
        dist.barrier()

    return val_metrics

def train(cfg):

    # Logger
    logger= get_logger(cfg)

    # Data
    if cfg.local_rank == 0:
        print("-"*10)
        print("DTYPE: {}".format(cfg.dataset_type))

    # Noisy volumes
    skip_tomo_ids= [
        "tomo_08bf73", "tomo_3a0914", "tomo_9f918e", "tomo_24a095", 
        "tomo_37c426", "tomo_692081", "tomo_b18127", "tomo_774aae", 
        "tomo_5b359d", "tomo_6b1fd3", "tomo_146de2", "tomo_9f424e", 
        "tomo_67ff4e", "tomo_ac4f0d", "tomo_d8c917", "tomo_f672c0", 
        "tomo_3b8291", "tomo_ab30af", "tomo_fc1665", "tomo_648adf", 
        "tomo_f36495", "tomo_e9fa5f", "tomo_fc90fd", "tomo_b28579", 
        "tomo_bd42fa", "tomo_c7a40f", "tomo_1c2534", "tomo_35ec84", 
        "tomo_ca1d13", "tomo_22976c", "tomo_2c9f35",
        ]

    # Datasets + Dataloaders
    df_train= pd.read_csv("./data/processed/folds_all.csv")
    df_train= df_train[~df_train["tomo_id"].isin(skip_tomo_ids)]
    df_train= df_train[df_train["fold"] != cfg.fold]
    if cfg.fast_dev_run or cfg.train == False: 
        df_train= df_train.head(cfg.batch_size)

    train_ds= get_dataset(df_train, cfg, mode="train")
    if cfg.world_size > 1:
        sampler= DistributedSampler(
            train_ds, 
            num_replicas=cfg.world_size, 
            rank=cfg.local_rank,
            )
    else:
        sampler= None
    train_dl= get_dataloader(train_ds, cfg, sampler=sampler, mode="train")

    df_val= pd.read_csv("./data/processed/folds_all.csv")
    df_val= df_val[~df_val["tomo_id"].isin(skip_tomo_ids)]
    df_val= df_val[df_val["fold"] == cfg.fold]
    if cfg.fast_dev_run: 
        df_val= df_val.head(cfg.batch_size)

    val_ds= get_dataset(df_val, cfg, mode="val")
    val_dl= get_dataloader(val_ds, cfg, mode="val")

    # Model + Optimizer + Scheduler
    model, emb_dim = get_model(cfg)
    if cfg.world_size > 1:
        model= model.to(cfg.local_rank)
        model= DistributedDataParallel(
            model, 
            device_ids=[cfg.local_rank], 
            )
    else:
        model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, n_steps=len(train_dl)*cfg.epochs)

    if cfg.mixed_precision:
        scaler = GradScaler(cfg.device.type)
    else:
        scaler = None

    # Training Loop
    train_metrics= {"train": {}, "lr": None, "epoch": None}
    val_metrics= {"val": {}}
    ema_model= None
    total_grad_norm = None    
    total_grad_norm_after_clip = None
    optimizer.zero_grad()
    i= 0
    epoch= 0
    time_epoch_start = time.time()

    if cfg.train:
        for epoch in range(cfg.epochs):
            time_epoch_prev = time.time() - time_epoch_start
            time_epoch_start = time.time()
            losses= []
            grad_norms= []
            grad_norms_clipped= []
            gc.collect()

            # Realod dataloaders if last iter took > 10 mins
            if time_epoch_prev > 60 * 10:
                train_dl= get_dataloader(train_ds, cfg, sampler=sampler, mode="train")
                val_dl= get_dataloader(val_ds, cfg, mode="val")

            if cfg.local_rank == 0: 
                train_metrics["epoch"] = epoch

            if cfg.local_rank == 0 and cfg.ema == True and epoch == 0:
                print("Starting EMA..")
                ema_model= ModelEMA(model, decay=cfg.ema_decay)

            if cfg.world_size > 1:
                train_dl.sampler.set_epoch(epoch)
                dist.barrier()

            progress_bar = tqdm(range(len(train_dl)), disable=cfg.local_rank!=0)
            tr_itr = iter(train_dl)

            # Iterate batches
            model.train()
            for itr in progress_bar:
                i += 1

                try:
                    batch= next(tr_itr)
                except Exception as e:
                    print(f"Batch failed: {e}")
                    continue

                if cfg.world_size > 1:
                    batch= batch_to_device(batch, device=cfg.local_rank)
                else:
                    batch= batch_to_device(batch, device=cfg.device)

                # Forward Pass
                if cfg.mixed_precision:
                    with autocast(cfg.device.type):
                        output= model(batch)
                else:
                    output= model(batch)
                loss= output["loss"]
                losses.append(loss.item())

                # Backward pass
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    if i % cfg.grad_accumulation == 0:
                        if (cfg.track_grad_norm) or (cfg.grad_clip > 0):
                            scaler.unscale_(optimizer)
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm is not None:
                                grad_norms.append(total_grad_norm.item())
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm_after_clip is not None:
                                grad_norms_clipped.append(total_grad_norm_after_clip.item())
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters())
                            if total_grad_norm is not None:
                                grad_norms.append(total_grad_norm.item())
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm_after_clip is not None:
                                grad_norms_clipped.append(total_grad_norm_after_clip.item())
                        optimizer.step()
                        optimizer.zero_grad() 

                if ema_model is not None:
                    ema_model.update(model)

                if scheduler is not None:
                    scheduler.step()
                    
                # Train Logging
                if cfg.local_rank == 0 and i % cfg.logging_steps == 0:
                    train_metrics["train"]["loss"]= np.mean(losses[-10:])
                    train_metrics["lr"]= cfg.lr if scheduler is None else scheduler.get_last_lr()[0]

                    if cfg.track_grad_norm:
                        train_metrics["grad_norm"] = np.mean(grad_norms[-10:])
                        train_metrics["grad_norm_clipped"] = np.mean(grad_norms_clipped[-10:])
            
                    # Log
                    progress_bar.set_postfix(flatten_dict(train_metrics | val_metrics))
                    logger.log(train_metrics, commit=False)
                    pass


            # Run eval
            if epoch != 0 and epoch % cfg.eval_epochs == 0: 
                if ema_model is not None:
                    val_metrics= run_eval(ema_model.module, val_ds, val_dl, val_metrics, cfg)
                else:
                    val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg)

            # Save weights
            if cfg.local_rank == 0 and epoch != 0 and cfg.save_weights and epoch % cfg.save_epochs == 0:
                if ema_model is not None:
                    save_weights(ema_model.module, cfg, epoch=epoch+1)
                else:
                    save_weights(model, cfg, epoch=epoch+1)

            # Log
            if cfg.local_rank == 0:
                progress_bar.set_postfix(flatten_dict(train_metrics | val_metrics))
                logger.log(val_metrics, commit=True)


    # Final eval
    if ema_model is not None:
        val_metrics= run_eval(ema_model.module, val_ds, val_dl, val_metrics, cfg)
    else:
        val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg)

    if cfg.local_rank == 0:
        logger.log(val_metrics)
        print(val_metrics)

    # Save weights
    if cfg.local_rank == 0 and cfg.save_weights:
        if ema_model is not None:
            save_weights(ema_model.module, cfg, epoch=epoch+1)
        else:
            save_weights(model, cfg, epoch=epoch+1)

    logger.finish()
    return
