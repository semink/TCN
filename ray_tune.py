import os
import sys
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch import nn

from TCN.low_resolution.model import LowResolutionTCN
from TCN.low_resolution.utils import get_traffic_data, TimeseriesDataset


def train(config, checkpoint_dir=None):
    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    model = LowResolutionTCN(input_size=config['input_dim'],
                             compress_dim=config['compress_dim'],
                             seq_length=config['seq_length'],
                             num_channels=[config['nhid']] * config['levels'],
                             kernel_size=config['kernel_size'],
                             dropout=config['dropout'])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Load dataset
    df_train, df_valid = get_traffic_data()
    train_dataset = TimeseriesDataset(df_train, seq_len=config['seq_length'])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=8)

    valid_dataset = TimeseriesDataset(df_valid, seq_len=config['seq_length'])
    # Load entire dataset for validation
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=8)

    for epoch in range(10):
        running_loss = 0.0
        epoch_steps = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.float().to(device), y.float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            # if args.clip > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        for i, (x, y) in enumerate(valid_loader):
            with torch.no_grad():
                x, y = x.float().to(device), y.float().to(device)

                output = model(x)
                total += y.size(0)
                loss = criterion(output, y)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {"input_dim": tune.choice([325]),
              "compress_dim": tune.sample_from(lambda _: 2 ** np.random.randint(1, 8)),
              "seq_length": tune.sample_from(lambda _: 2 ** np.random.randint(1, 8)),
              "nhid": tune.sample_from(lambda _: 2 ** np.random.randint(3, 7)),
              "levels": tune.sample_from(lambda _: 2 ** np.random.randint(1, 4)),
              "kernel_size": tune.sample_from(lambda _: 2 ** np.random.randint(1, 5)),
              "dropout": tune.choice([0, 0.1, 0.5]),
              "lr": tune.loguniform(1e-4, 1e-1),
              "batch_size": tune.choice([8, 16, 32, 64, 128])}
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns=["compress_dim", "seq_length", "nhid", "levels",
                           "kernel_size", "dropout", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"]
    )
    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    # best_trained_model = LowResolutionTCN(input_size=best_trial.config['input_dim'],
    #                                       compress_dim=best_trial.config['compress_dim'],
    #                                       seq_length=best_trial.config['seq_length'],
    #                                       num_channels=best_trial.config['nhid'] * best_trial.config['levels'],
    #                                       kernel_size=best_trial.config['kernel_size'],
    #                                       dropout=best_trial.config['dropout'])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)
    #
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=2)
