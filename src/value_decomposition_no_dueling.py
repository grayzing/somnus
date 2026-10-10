"""VDN ablation: same as vdn.py but the agent network uses a plain linear Q-head (no dueling)."""
from qmix import train

if __name__ == "__main__":
    train("vdn", dueling=False)