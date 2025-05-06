import os
import logging
import shutil
import datetime
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import wandb
from tqdm import tqdm 

from models.ensemble import MaskedDynamicsEnsemble
from data.dataset import load_pusht_dataset 

def mse_loss(pred, target, gate):
    loss = ((pred - target) ** 2) * gate
    return loss.sum() / gate.sum().clamp_min(1)

def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def make_random_mask(state_hist, max_p, epoch, warmup, start=10):
    """
    Curriculum masking: start at 0% until start epochs, ramp to max_p by 'warmup' epochs afterwards.
    """
    p = min(max_p, max_p * max(epoch - start, 0) / warmup)
    rand = torch.rand_like(state_hist[..., 0])          # (B,H,D)
    return (rand > p).unsqueeze(1).expand(-1, state_hist.shape[1], -1)    # True = observed


def main(config_path: str):
    # Load configuration
    cfg = OmegaConf.load(config_path)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.train.no_cuda else 'cpu')

    # Create a per-run checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # If using WandB, include the run name; otherwise just timestamp
    run_id = cfg.wandb.run_name if (cfg.wandb.enable and cfg.wandb.run_name) else timestamp
    run_dir = os.path.join(cfg.train.checkpoint_dir, f"{run_id}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Copy original config YAML into the run directory
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    # Write a run_info.txt with timestamp and full resolved config
    with open(os.path.join(run_dir, "run_info.txt"), "w") as info_f:
        info_f.write(f"Run timestamp: {timestamp}\n\n")
        info_f.write("Full resolved configuration:\n")
        info_f.write(OmegaConf.to_yaml(cfg))

    # Logger (points to a file inside this run directory)
    log_file = os.path.join(run_dir, "train.log")
    logger = setup_logger(log_file)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # WandB init (optional)
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load data
    train_ds, val_ds = load_pusht_dataset(cfg.data)
    train_loaders = []
    val_loader    = DataLoader(  # (shared validation set)
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers)

    subset_datasets = []
    for m in range(cfg.model.num_models):
        # draw with replacement → bootstrap sample, but draw ONCE
        boot_idx = torch.randint(
            high=len(train_ds),
            size=(int(cfg.data.train_frac * len(train_ds)),),
            generator=torch.Generator().manual_seed(cfg.seed + m)  # reproducible
        )
        subset_datasets.append(Subset(train_ds, boot_idx))
    train_loaders = [
        DataLoader(
            ds,
            batch_size   = cfg.train.batch_size,
            shuffle      = True,      # order reshuffles every __iter__ (=epoch)
            num_workers  = cfg.train.num_workers,
            pin_memory   = True
        )
        for ds in subset_datasets
    ]

    # Model and optimizer
    model = MaskedDynamicsEnsemble(cfg).to(device)
    optims = [
        torch.optim.Adam(model.models[m].parameters(),
                        lr=cfg.train.learning_rate)
        for m in range(cfg.model.num_models)
    ]

    best_val_loss = float('inf')
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0
        iters = [iter(dl) for dl in train_loaders]
        steps = min(len(dl) for dl in train_loaders)   # same #steps for fairness

        for step in tqdm(range(steps)):
            for m_idx, it_ in enumerate(iters):
                batch = next(it_)
                s_hist = batch['state'].to(device)
                a_hist = batch['action'].to(device)
                s_next = batch['next_state'].to(device)

                mask_hist = make_random_mask(
                    s_hist, cfg.train.max_mask_prob,
                    epoch, cfg.train.mask_warmup_epochs).to(device)

                optims[m_idx].zero_grad()
                pred = model.models[m_idx](s_hist, a_hist, mask_hist)  # single net
                loss = mse_loss(pred, s_next, mask_hist[:, -1])  
                loss.backward()
                optims[m_idx].step()

                total_loss += loss.item()

        avg_train_loss = total_loss / (steps * cfg.model.num_models)
        logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}")
        if cfg.wandb.enable:
            wandb.log({"train/loss": avg_train_loss}, step=epoch)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                states, actions = batch['state'], batch['action']
                mask = torch.ones_like(states, dtype=torch.bool)
                preds, _ = model(
                    states.to(device), actions.to(device), mask.to(device)
                )
                val_loss += torch.nn.functional.mse_loss(
                    preds, batch['next_state'].to(device)
                ).item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch}, Val Loss: {avg_val_loss:.6f}")
        if cfg.wandb.enable:
            wandb.log({"val/loss": avg_val_loss}, step=epoch)

        # ---- Checkpoint ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(run_dir, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
