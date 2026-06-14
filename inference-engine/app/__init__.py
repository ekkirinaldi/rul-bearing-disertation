"""RUL streaming inference engine (FastAPI + WebSocket).

Wraps the dissertation's trained Mamba-xLSTM-Net checkpoints behind a small
streaming API: per-acquisition HI extraction, live RUL prediction, fusion
branch-gate balance, and on-demand Integrated Gradients explanations.
"""
