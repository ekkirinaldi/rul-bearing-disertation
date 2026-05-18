"""Journal Q2 (JETS) extra experiments package.

This package collects scripts that produce the additional evidence
required by the JETS paper "Mechanistic Interpretability of Bearing
Remaining Useful Life Models via Top-k Sparse Autoencoders":

  * statistical inference   — bootstrap CI + permutation tests
  * negative controls       — untrained backbone + Gaussian noise
  * sparsity sweep          — k in {10, 51, 102, 205}
  * cross-architecture      — N-BEATS-xLSTM and SparseGate-TCN

Outputs are written to ``Mamba-xLSTM/results/journal_q2/<script>/``
and consumed by the journal paper at
``writings/journal-q2/jets-mechanistic-interp/``.
"""
