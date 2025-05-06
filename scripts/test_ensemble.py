# scripts/test_ensemble.py
"""
Quick integration test for the MaskedDynamicsEnsemble.
Usage:
    python scripts/test_ensemble.py
"""
import sys
import torch

# Add src to path
sys.path.append('src')

from models.ensemble import MaskedDynamicsEnsemble

# Minimal config namespace
class ModelCfg: pass
class Cfg: pass

cfg = Cfg()
cfg.model = ModelCfg()
cfg.data = ModelCfg()
cfg.model.num_models = 5
cfg.model.state_dim = 8
cfg.model.action_dim = 2
cfg.model.embedding_dim = 4
cfg.model.n_heads = 4
cfg.model.n_layers = 2
cfg.model.feedforward_dim = 64
cfg.model.dropout = 0.1
cfg.data.num_hist = 3

# Instantiate ensemble
ensemble = MaskedDynamicsEnsemble(cfg)
print("Ensemble instantiated with", cfg.model.num_models, "members.")

# Dummy data
batch_size = 2
state = torch.randn(batch_size, cfg.data.num_hist, cfg.model.state_dim)
action = torch.randn(batch_size, cfg.data.num_hist, cfg.model.action_dim)
mask = torch.ones(batch_size, cfg.data.num_hist, cfg.model.state_dim, dtype=torch.bool)

# Forward
mu, var = ensemble(state, action, mask)
print("Forward output shapes:", mu.shape, var.shape)

# Rollout
seq_len = 3
actions_seq = torch.randn(batch_size, seq_len, cfg.model.action_dim)
states_traj, masks_traj = ensemble.rollout_with_dropout(state, action, mask, actions_seq, var_threshold=0.5)
print("Rollout shapes:", states_traj.shape, masks_traj.shape)
