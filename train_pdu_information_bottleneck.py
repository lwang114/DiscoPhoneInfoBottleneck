"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import editdistance
import itertools
import json
import logging
import os
import sys
import time
import torch
import torch.nn as nn
import models
import utils
from pdu_information_bottlenecks import PositionDependentUnigramBottleneck
from evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a visually-grounded phoneme discovery model."
    )
    parser.add_argument(
        "--config", type=str, help="A json configuration file for experiment."
    )
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--restore", action="store_true", help="Restore training from last checkpoint")
    parser.add_argument("--last_epoch", type=int, default=0, help="Epoch restoring from.")
    parser.add_argument(
        "--checkpoint_path",
        default="/tmp/",
        type=str,
        help="Checkpoint path for saving models",
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="world size for distributed training"
    )
    parser.add_argument(
        "--eval_only", action="store_true"
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    use_cpu = args.disable_cuda or not torch.cuda.is_available()

    if args.restore:
        logging.info(f"Restoring model from epoch {args.last_epoch}")

    return args


@torch.no_grad()
def test(model, data_loader, device, checkpoint_path='./'):
    model.eval()
    pred_dicts = []
    pooling_ratio = None
    with torch.no_grad():
      total = 0.
      correct = 0.
      for b_idx, (inputs, targets, input_masks) in enumerate(data_loader):
        if b_idx == 0:
            B = inputs.size(0)
        in_scores, out_scores = model(inputs.to(device), input_masks.to(device))

        pred_label = out_scores.topk(1, dim=-1)[-1].detach().cpu().numpy()
        gold_label = targets.detach().cpu().numpy()
        correct += (pred_label == gold_label).sum()
        total += targets.size(0)
        
        prediction = out_scores.topk(1, dim=-1)[-1].squeeze(-1)
        prediction = prediction.cpu().detach().numpy().tolist()
        
        for idx in range(inputs.size(0)):
            global_idx = b_idx * B + idx
            example_id = data_loader.dataset.dataset[global_idx][0]
            text = data_loader.dataset.dataset[global_idx][1]
            nframes = int(input_masks[idx].sum().cpu().detach().numpy())
            in_scores_ds = in_scores[idx].cpu().detach().numpy()
            in_scores_ds = in_scores_ds.reshape(8, int(in_scores[idx].size(0) // 8), -1).mean(axis=0)
            nframes_ds = int(nframes // 8)
            pred_dicts.append(
                {'sent_id': example_id,
                 'units': prediction[idx],
                 'scores': in_scores_ds[:nframes_ds].tolist(),
                 'nframes': nframes,
                 'text': text})

    acc = correct / total
    
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    json.dump(pred_dicts, open(os.path.join(checkpoint_path, 'predictions.json'), 'w'), indent=2)
    return acc

def checkpoint(model, checkpoint_path, save_best=False):
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    model_checkpoint = os.path.join(checkpoint_path, "model.checkpoint")
    # criterion_checkpoint = os.path.join(checkpoint_path, "criterion.checkpoint")
    torch.save(model.state_dict(), model_checkpoint)
    # torch.save(criterion.state_dict(), criterion_checkpoint)
    if save_best:
        torch.save(model.state_dict(), model_checkpoint + ".best")
        # torch.save(criterion.state_dict(), criterion_checkpoint + ".best")

def train(args, world_rank=0):
    # setup logging
    level = logging.INFO
    logging.getLogger().setLevel(level)

    if not args.disable_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with open(args.config, "r") as fid:
        config = json.load(fid)
        logging.info("Using the config \n{}".format(json.dumps(config)))

    # seed everything:
    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    # setup data loaders:
    logging.info("Loading dataset ...")
    dataset = config["data"]["dataset"]
    if not os.path.exists(f"datasets/{dataset}.py"):
        raise ValueError(f"Unknown dataset {dataset}")
    dataset = utils.module_from_file("dataset", f"datasets/{dataset}.py")

    input_size = config["data"]["num_features"]
    data_path = config["data"]["data_path"]
    batch_size = config["optim"]["batch_size"]

    preprocessor = dataset.Preprocessor(
        data_path,
        num_features=input_size,
        tokens_path=config["data"].get("tokens", None),
        lexicon_path=config["data"].get("lexicon", None),
        use_words=config["data"].get("use_words", False),
        prepend_wordsep=config["data"].get("prepend_wordsep", False),
        supervised=config["data"].get("supervised", False),
        level=config["data"].get("level", "phone") 
    )    
    output_size = preprocessor.num_tokens
    trainset = dataset.Dataset(data_path, preprocessor, split="train", augment=True)
    valset = dataset.Dataset(data_path, preprocessor, split="test", augment=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # setup criterion, model:
    logging.info("Loading model ...")
    criterion = nn.CrossEntropyLoss()
    
    model = PositionDependentUnigramBottleneck(input_size, output_size,config["model_type"], **config['model'])
        
    if args.restore:
        model_checkpoint = os.path.join(args.checkpoint_path, "model.checkpoint")
        model.load_state_dict(torch.load(model_checkpoint))

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(
        "Training {} model with {:,} parameters.".format(config["model_type"], n_params)
    )

    # Store base module, criterion for saving checkpoints
    base_model = model
    if not isinstance(model, torch.nn.DataParallel):
      model = nn.DataParallel(model)

    epochs = config["optim"]["epochs"]
    lr = config["optim"]["learning_rate"]
    step_size = config["optim"]["step_size"]
    max_grad_norm = config["optim"].get("max_grad_norm", None)

    # run training:
    logging.info("Starting training ...")
    scale = 0.5 ** (args.last_epoch // step_size)
    params = [{"params" : model.parameters(),
               "initial_lr" : lr * scale,
               "lr" : lr * scale}]

    optimizer = torch.optim.SGD(params)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=0.5,
        last_epoch=args.last_epoch,
    )
    max_val_acc = float("-inf")

    Timer = utils.CudaTimer if device.type == "cuda" else utils.Timer
    timers = Timer(
        [
            "ds_fetch",  # dataset sample fetch
            "model_fwd",  # model forward
            "crit_fwd",  # criterion forward
            "bwd",  # backward (model + criterion)
            "optim",  # optimizer step
            "metrics",  # viterbi, cer
            "train_total",  # total training
            "test_total",  # total testing
        ]
    )
    num_updates = 0
    for epoch in range(args.last_epoch, epochs):
        logging.info("Epoch {} started. ".format(epoch + 1))
        model.train()
        start_time = time.time()
        meters = utils.IBMeters()
        
        timers.reset()
        timers.start("train_total").start("ds_fetch")
        for i, (inputs, targets, input_masks) in enumerate(train_loader):
            if args.eval_only:
                continue
            timers.stop("ds_fetch").start("model_fwd")
            optimizer.zero_grad()
            in_scores, trg_scores = model(inputs.to(device), input_masks)
            prediction_loss = criterion(trg_scores, targets.to(device))
            loss, I_ZX, I_ZY = model.module.calculate_loss(in_scores, prediction_loss)
            
            timers.stop("model_fwd").start("crit_fwd")
            timers.stop("crit_fwd").start("bwd")
            loss.backward()
            timers.stop("bwd").start("optim")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )
            optimizer.step()
            num_updates += 1
            timers.stop("optim").start("metrics")
            meters.loss += loss.item() * len(targets)
            meters.num_samples += len(targets)
            meters.I_ZX += I_ZX.item() * len(targets)
            meters.I_ZY += I_ZY.item() * len(targets)
            meters.num_tokens += len(targets)
            timers.stop("metrics").start("ds_fetch")
            if i % 1000 == 0:
                info = 'Itr {} {meters.loss:.3f} ({meters.avg_loss:.3f})'.format(i, meters=meters)
                print(info)
        timers.stop("ds_fetch").stop("train_total")
        epoch_time = time.time() - start_time

        logging.info(
            "Epoch {} complete. "
            "nUpdates {}, Loss {:.3f} "
            " Time {:.3f} (s), LR {:.3f}".format(
                epoch + 1,
                num_updates,
                meters.avg_loss,
                epoch_time,
                scheduler.get_last_lr()[0],
            ),
        )

        if epoch % 1 == 0:
          logging.info("Evaluating validation set..")
          timers.start("test_total")
          val_acc = test(
             model, val_loader, device, args.checkpoint_path
          )
          
          timers.stop("test_total")
          checkpoint(
                  base_model,
                  args.checkpoint_path,
                  (val_acc > max_val_acc),
          )

          max_val_acc = max(val_acc, max_val_acc) 
          logging.info(
              "Validation Set: WER {:.3f}, "
              "Best WER {:.3f}".format(
                1-val_acc, 1-max_val_acc
            ),
          )
          logging.info(
              "Timing Info: "
              + ", ".join(
                  [
                      "{} : {:.2f}ms".format(k, v * 1000.0)
                      for k, v in timers.value().items()
                  ]
              )
          )
        scheduler.step()
        start_time = time.time()

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
