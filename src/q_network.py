import torch
from torch import nn


class DeepSetQ(nn.Module):
    """
    Per-agent (per-base-station) Q-network over a *set* of UEs (dueling heads by default; `dueling=False` for the ablation).

    The set for an agent is every UE currently attached to one of the agent's
    neighbourhood sectors (own sectors + neighbouring sectors). Each UE is a
    vector of KPIs. The network is permutation-invariant and handles a variable
    number of UEs via padding + mask:

        phi(ue_i)  -> per-UE embedding                (shared weights)
        pool       -> masked mean || masked max || log(|set|)
        rho([pool, sector context]) -> dueling heads (or a single linear head) -> Q(s, a) for a in [0, num_actions)

    Shapes:
        ue_feats : (B, M, F)   padded UE features (padding rows are ignored)
        mask     : (B, M)      bool, True for real UEs
        context  : (B, C)      fixed-size per-agent context (sleep modes / UE counts per sector)
        returns  : (B, num_actions)
    """

    def __init__(self, ue_feat_dim: int, context_dim: int, num_actions: int = 12,
                 phi_dim: int = 128, rho_dim: int = 256, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.phi = nn.Sequential(
            nn.Linear(ue_feat_dim, phi_dim), nn.ReLU(),
            nn.Linear(phi_dim, phi_dim), nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * phi_dim + 1 + context_dim, rho_dim), nn.ReLU(),
            nn.Linear(rho_dim, rho_dim), nn.ReLU(),
        )
        if dueling:
            self.value_stream = nn.Linear(rho_dim, 1)
            self.advantage_stream = nn.Linear(rho_dim, num_actions)
        else:
            # Ablation: identical trunk, a single linear head straight to Q-values
            self.q_head = nn.Linear(rho_dim, num_actions)

    @staticmethod
    def masked_pool(h: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(-1)                                   # (B, M, 1)
        count = mask.sum(dim=1, keepdim=True).to(h.dtype)        # (B, 1)
        mean = (h * m).sum(dim=1) / count.clamp(min=1.0)
        mx = h.masked_fill(~m, -1e9).max(dim=1).values
        mx = torch.where(count > 0, mx, torch.zeros_like(mx))    # empty set -> zeros
        return mean, mx, count

    def forward(self, ue_feats: torch.Tensor, mask: torch.Tensor, context: torch.Tensor):
        h = self.phi(ue_feats)                                   # (B, M, H)
        mean, mx, count = self.masked_pool(h, mask)
        z = torch.cat([mean, mx, torch.log1p(count) / 5.0, context], dim=1)
        z = self.rho(z)

        if not self.dueling:
            return self.q_head(z)
        value = self.value_stream(z)
        advantage = self.advantage_stream(z)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class VDNMixer(nn.Module):
    """Value Decomposition Network: Q_tot = sum_i Q_i. The state is ignored."""

    def forward(self, agent_qs: torch.Tensor, state=None) -> torch.Tensor:
        return agent_qs.sum(dim=1, keepdim=True)                 # (B, 1)


class QMixer(nn.Module):
    """
    QMIX monotonic mixing network. The global state is a compact fixed-size
    per-sector summary (sleep mode, UE count, mean throughput, SLA fraction,
    mean SNR), not the concatenation of every agent's raw observation.
    """

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32, hyper_hidden: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, num_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = agent_qs.shape[0]
        qs = agent_qs.view(B, 1, self.num_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(B, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        hidden = nn.functional.elu(torch.bmm(qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(B, 1)