"""
Independent Q-Learning (IQL) baseline.

Same DeepSet agent network, observations, replay buffer and shared team reward as QMIX/VDN, but there is
no mixing network: every agent is trained on its own TD error,
    y_i = r_team + gamma * (1 - done) * max_a' Q_target_i(o'_i, a')
    L   = mean_i SmoothL1(Q_i(o_i, a_i), y_i)
so credit assignment between agents is not attempted.
"""
from qmix import train

if __name__ == "__main__":
    train("iql")