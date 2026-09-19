# Compute Setup

How the prognostic runs in `research/Mamba-xLSTM/` were set up on rented GPU machines (RunPod pods and plain CUDA VPS hosts). The laptop is used for editing and small checks; anything longer than a smoke test runs remotely.

## Directory layout on the remote

The data configs refer to the datasets as `../data-bearing/…`, so `data-bearing/` must sit next to `Mamba-xLSTM/` under one parent directory, as it does under `research/` in this repository:

```
<parent>/
  Mamba-xLSTM/
  data-bearing/
```

## SSH access

Generate a key for the host on the laptop and append the public half to the server's `authorized_keys`:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod -C "label-for-this-host"
cat ~/.ssh/id_ed25519_runpod.pub          # copy the single line
```

On the server:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat >> ~/.ssh/authorized_keys << 'EOF'
<paste the public key line>
EOF
chmod 600 ~/.ssh/authorized_keys
```

Connect with the host, port and user the provider gives (RunPod usually uses a port other than 22):

```bash
ssh -i ~/.ssh/id_ed25519_runpod -p <PORT> <USER>@<HOST>
```

If login still fails, check that `~/.ssh` is mode 700 and `authorized_keys` is mode 600, that the key is on a single line, and run `ssh -v` to see which key is offered. RunPod pods lose their home directory when recreated, so the key has to be added again after a redeploy.

## Datasets

On a new machine the datasets are downloaded from S3, not copied from the laptop. Rsync of the full tree is slow and tends to drop on long links. All archives are public objects in the same bucket:

```bash
cd <parent>
B=https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing

curl -fL -o data-bearing.zip "$B/data-bearing.zip" && unzip -q data-bearing.zip && rm data-bearing.zip

# XJTU-SY: the base archive carries an incomplete XJTU tree; this one replaces it
curl -fL -o xtju-sy.zip "$B/xtju-sy.zip" && unzip -q -o xtju-sy.zip -d data-bearing/ && rm xtju-sy.zip

# IMS (optional): expands to IMS/, which belongs under data-bearing/
curl -fL -o IMS.zip "$B/IMS.zip" && unzip -q IMS.zip && rm IMS.zip && mv IMS data-bearing/
```

The CWRU mirror (`$B/cwru.zip`) may not be published. Check it with `curl -fIL` before relying on it; otherwise download the `.mat` files from the Case Western Reserve University Bearing Data Center into `data-bearing/cwru/`.

Before a long comparison, check that the data is complete:

```bash
test -d data-bearing/xtju-sy/35Hz12kN && test -d data-bearing/xtju-sy/37.5Hz11kN && test -d data-bearing/xtju-sy/40Hz10kN
find data-bearing/xtju-sy -name '*.csv' | wc -l          # 9216
ls data-bearing/ieee-phm-2012/Learning_set/Bearing1_1 | head -2
```

Minimal images may lack `curl`, `unzip` or `rsync`; install them with `apt-get install -y curl unzip rsync`.

## Code and environment

Only the code travels from the laptop. `scripts/rsync_training_bundle_to_vps.sh` skips `.venv`, caches and the results tree:

```bash
rsync -az --delete -e "ssh -i ~/.ssh/id_ed25519_runpod -p <PORT>" \
  research/Mamba-xLSTM/ <USER>@<HOST>:<parent>/Mamba-xLSTM/
```

On the server, `scripts/bootstrap_gpu_vps.sh` creates the virtual environment with CUDA wheels. It defaults to `TORCH_CUDA=cu128`, which matches hosts that report CUDA 12.8 in `nvidia-smi`; `INSTALL_MAMBASSM=1` also builds the `mamba_ssm` kernels, which takes a while. Training uses a single GPU and picks CUDA automatically.

A smoke run after the bootstrap:

```bash
cd Mamba-xLSTM && source .venv/bin/activate
python scripts/train.py --data configs/data/phm2012.yaml \
  --model configs/model/nbeats_rul.yaml \
  --train configs/train/algorithm_comparison.yaml \
  --ablation configs/ablation/gpu_throughput.yaml \
  --fast-dev-run
```

## Training presets

| Preset | Use |
|---|---|
| `configs/train/algorithm_comparison.yaml` | quick comparisons |
| `configs/train/cloud_full_75.yaml` | 75 epochs, `bf16-mixed`; `run_algorithm_comparison.py --mode cloud` |
| `configs/train/full_run.yaml` | 200 epochs, fp32; `--mode full` |
| `configs/ablation/gpu_throughput.yaml` | `num_workers: 8`, merged with `--ablation` |

Small HI windows rarely saturate a GPU on their own. Raise the batch size with `--data-batch-size N` until `nvidia-smi` shows steady utilisation without running out of memory.

Long jobs run under `nohup` and log to the home directory. The `wait_pull_*` scripts poll that log every 15 minutes and pull `results/` back once the report line appears. `VPS_HOST`, `VPS_PORT`, `VPS_KEY` and `REMOTE_LOG` override their defaults.

```bash
nohup python -u scripts/run_algorithm_comparison.py \
  --datasets phm2012 xjtusy --models phase_moe_xlstm_rul sparse_gate_tcn_rul \
  --train configs/train/cloud_full_75.yaml --ablation configs/ablation/gpu_throughput.yaml \
  --data-batch-size 512 --report-name phase_sparse_cloud75_s42 > ~/vps_phase_sparse_75ep.log 2>&1 &
```
