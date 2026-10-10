"""
VDN training: identical DeepSet agents, observations, replay and reward as QMIX,
but the mixer is a plain sum of per-agent Q-values (Q_tot = sum_i Q_i).
"""
from qmix import train

if __name__ == "__main__":
    train("vdn")