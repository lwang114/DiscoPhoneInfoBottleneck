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

import models
import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a handwriting recognition model."
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
        "--dist_url",
        default="tcp://localhost:23146",
        type=str,
        help="url used to set up distributed training. This should be"
        "the IP address and open port number of the master node",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    use_cpu = args.disable_cuda or not torch.cuda.is_available()
    if args.world_size > 1 and use_cpu:
        logging.fatal("CPU distributed training not supported.")
        sys.exit(1)

    logging.info("World size is : " + str(args.world_size))
    if args.restore:
        logging.info(f"Restoring model from epoch {args.last_epoch}")

    if not use_cpu and torch.cuda.device_count() < args.world_size:
        logging.fatal(
            "At least {} cuda devices required. {} found".format(
                args.world_size, torch.cuda.device_count()
            )
        )
        sys.exit(1)

    return args


@torch.no_grad()
def test(model, data_loader, checkpoint_path='./'):
    model.eval()
    meters = utils.Meters()
    # predictions = []

    I_outputs = []
    A_outputs = []
    I_embeddings = []
    A_embeddings = []
    with torch.no_grad():
      for b_idx, (inputs, targets) in enumerate(data_loader):
        if b_idx == 0:
          batch_size = inputs.size(0)
        
        outputs = model(inputs.to(device))
        meters.loss += model.module.calculate_loss(outputs, targets).item() * len(targets)
        meters.num_samples += len(targets)
        tokens_dist, words_dist, n_tokens, n_words = compute_edit_distance(
            criterion.viterbi(outputs), targets, preprocessor
        )
        meters.edit_distance_tokens += tokens_dist
        meters.num_tokens += n_tokens
        meters.edit_distance_words += words_dist
        meters.num_words += n_words

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    return meter.avg_loss, recalls

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


def train(args):
    # setup logging
    level = logging.INFO
    logging.getLogger().setLevel(level)

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
    output_size = config["data"]["num_visual_features"]
    data_path = config["data"]["data_path"]
    batch_size = config["data"]["batch_size"]
    

    trainset = dataset.Dataset(data_path, split="train") # TODO
    valset = dataset.Dataset(data_path, split="validation")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # setup criterion, model:
    logging.info("Loading model ...")
    model = AudioVisualInformationBottleneck(input_size, output_size, **config['model'])

    # if args.restore: # TODO
    #     load_from_checkpoint(model, args.checkpoint_path, True)
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
    min_val_loss = float("inf")
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
        meters = utils.Meters()
        # TODO Token F1 meter
        timers.reset()
        timers.start("train_total").start("ds_fetch")
        for inputs, targets in train_loader:
            timers.stop("ds_fetch").start("model_fwd")
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            timers.stop("model_fwd").start("crit_fwd")
            loss = model.module.calculate_loss(outputs, targets)
            timers.stop("crit_fwd").start("bwd")
            loss.backward()
            timers.stop("bwd").start("optim")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(model.parameters(), criterion.parameters()),
                    max_grad_norm,
                )
            optimizer.step()
            num_updates += 1
            timers.stop("optim").start("metrics")
            meters.loss += loss.item() * len(targets)
            meters.num_samples += len(targets)
            timers.stop("metrics").start("ds_fetch")
        timers.stop("ds_fetch").stop("train_total")
        epoch_time = time.time() - start_time
        if args.world_size > 1:
            meters.sync()
        logging.info(
            "Epoch {} complete. "
            "nUpdates {}, Loss {:.3f},"
            " Time {:.3f} (s), LR {:.3f}".format(
                epoch + 1,
                num_updates,
                meters.avg_loss,
                epoch_time,
                scheduler.get_last_lr()[0],
            ),
        )
        if epoch % 5 == 0:
          logging.info("Evaluating validation set..")
          timers.start("test_total")
          # TODO
          val_loss, recalls = test(
             model, val_loader, args.checkpoint_path
          )
          val_acc = (recalls['A_r10'] + recalls['I_r10']) / 2

          timers.stop("test_total")
          checkpoint(
                  base_model,
                  base_criterion,
                  args.checkpoint_path,
                  (val_acc > max_val_acc),
          )

          min_val_loss = min(val_loss, min_val_loss)
          max_val_acc = max(val_acc, max_val_acc) 
          logging.info(
            "Validation Set: Loss {:.3f}, A2I R@1 {:.3f}, R@5 {:.3f}, R@10 {:.3f}, I2A R@1 {:.3f}, R@5 {:.3f}, R@10 {:.3f}"
            "Best Loss {:.3f}, Best Avg R@10 {:.3f}".format(
                val_loss, recalls['A_r1'], recalls['A_r5'], recalls['A_r10'], recalls['I_r1'], recalls['I_r5'], recalls['I_r10'], min_val_loss, max_val_acc
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
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            train, args=(args,), nprocs=args.world_size, join=True
        )
    else:
        train(0, args)


if __name__ == "__main__":
    main()
