# Frozen run snapshots (historical)

YAML files here are **merged configs from past training runs**, not the live
source of truth.

**Current dissertation XJTU-SY split** — edit only:

`configs/data/xjtu_sy_available_full.yaml`

(9 train / 3 val / 3 test; all three conditions `35Hz12kN`, `37.5Hz11kN`, `40Hz10kN`.)

Snapshots under `xjtusy_*_merged.yaml` still reflect the **old 2-condition**
split (5 train / 1 val / 2 test) unless regenerated after a new
`run_algorithm_comparison.py` job.
