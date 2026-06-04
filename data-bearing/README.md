# `data-bearing/` — Bearing Datasets (Not Tracked in Git)

The contents of this directory are **excluded from the repository** because the
combined raw + processed data is roughly **8.5 GB**. To reproduce experiments,
download the datasets into this folder using one of the options below.

Expected layout after download:

```
data-bearing/
├── ieee-phm-2012/        # PHM2012 / FEMTO-PRONOSTIA
│   ├── Learning_set/
│   ├── Test_set/
│   └── Full_Test_Set/
├── xtju-sy/              # XJTU-SY (note the on-disk spelling)
│   ├── 35Hz12kN/
│   ├── 37.5Hz11kN/
│   └── 40Hz10kN/
├── IMS/                  # IMS Bearing Dataset (optional)
├── MFPT Fault Data Sets/ # MFPT (optional)
├── skf-ch15-or1/         # PT SKF Observer exports — CH-15 OR-1 (local/industrial)
└── processed/            # Derived parquet caches (regenerable from raw)
```

The training/eval pipeline expects this exact layout — configs reference paths
such as `../data-bearing/ieee-phm-2012/Learning_set/Bearing*/acc_*.csv` and
`../data-bearing/xtju-sy/<condition>/Bearing*/*.csv`.

---

## Option A — S3 canonical mirror (recommended, fastest)

A pre-zipped bundle of the full `data-bearing/` tree is hosted on S3:

```
https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/data-bearing.zip
```

From the **repository root** (parent of this directory):

```bash
curl -fL -o data-bearing.zip \
  'https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/data-bearing.zip'
unzip -q data-bearing.zip
rm -f data-bearing.zip
```

Use `wget -O data-bearing.zip '<URL>'` if `curl` is unavailable. On minimal
Debian/Ubuntu images: `apt-get update && apt-get install -y curl unzip`.

After unzipping, `data-bearing/` should sit beside `Mamba-xLSTM/` so configs
that use `../data-bearing/...` resolve correctly.

---

## Option B — Original sources (for verification or partial download)

### PHM2012 / FEMTO-PRONOSTIA

- **Owner:** FEMTO-ST Institute (Université de Franche-Comté)
- **Page:** <https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge.php>
- **Mirror (IEEE DataPort):** <https://ieee-dataport.org/open-access/phm-2012-data-challenge>
- **Citation:** Nectoux et al. (2012), *PRONOSTIA: An experimental platform for bearings accelerated degradation tests.*
- **Test rig:** PRONOSTIA, NSK 6804 deep-groove ball bearings.
- **Sampling:** 25.6 kHz, 0.1 s per record, 10 s interval, 17 bearings (6 train + 11 test) across 3 operating conditions.

After download, place the `Learning_set/`, `Test_set/`, and `Full_Test_Set/`
folders under `data-bearing/ieee-phm-2012/`.

### XJTU-SY Bearing Datasets

- **Owners:** Xi'an Jiaotong University & Sumyoung Technology Co., Ltd.
- **Page:** <https://biaowang.tech/xjtu-sy-bearing-datasets/>
- **Mirror (IEEE DataPort):** <https://ieee-dataport.org/open-access/xjtu-sy-bearing-datasets>
- **Citation:** Wang et al. (2020), *A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings.* IEEE TR.
- **Test rig:** XJTU-SY accelerated life test, LDK UER204 deep-groove ball bearings.
- **Sampling:** 25.6 kHz, 1.28 s per record, 1 min interval, 15 bearings across 3 operating conditions; failure modes labelled.

After download, place the per-condition folders under `data-bearing/xtju-sy/`
(note the directory name uses the `xtju-sy` spelling for compatibility with
existing configs).

### IMS Bearing Dataset (optional)

- **Page (NASA Prognostics Data Repository):** <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Place under `data-bearing/IMS/`.

### MFPT Fault Data Sets (optional)

- **Page:** <https://www.mfpt.org/fault-data-sets/>
- Place under `data-bearing/MFPT Fault Data Sets/`.

---

## Notes

- **`processed/`** can always be regenerated from the raw data by the training
  pipeline; it is not required to download it separately if you have the raw
  PHM2012 / XJTU-SY trees.
- **`data/cache/`** at the repository root is also git-ignored; it is rebuilt
  on demand by the data loaders.
- Do **not** commit any files under `data-bearing/` other than this `README.md`.
  The root `.gitignore` enforces this.
