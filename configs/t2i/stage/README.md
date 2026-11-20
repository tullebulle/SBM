# The text-to-image recipe, stage by stage

The released text-to-image models were trained in three stages, with checkpoint averages in between:

1. **CC12M pre-training** (`configs/t2i/exp/t2i_rae_xl1dh.yaml`, `t2i_rae_b1dh_t15.yaml`): 1M steps from scratch.
2. **The 120k alignment mix** (`+stage=mix`, [mix.yaml](mix.yaml)): 160k steps, the whole model.
3. **The DDT head alone** (`+stage=headonly`, [headonly.yaml](headonly.yaml)) on CC12M and the mix together, a few
   thousand steps.

Every later stage continues the step counter, optimizer and loss state of its starting checkpoint, and that starting point
is a checkpoint average (`scripts/t2i/ckpt/`):

- `soup.py` averages each EMA copy over a run's checkpoints (uniform, or weighted with `--weights=`);
- `nohead.py` keeps an average's trunk and embedders and takes the DDT head from another checkpoint or average;
- `soup_to_resume.py` writes the EMA-0.9999 copy of the result into every weight slot of a full checkpoint of the run
  (its optimizer, loss state and step), so the next stage resumes from it. Without `--soup` it fills the slots with the
  checkpoint's own EMA-0.9999 copy.

The averages and the stopping points were picked by GenEval. Global batch 256 throughout (16 GPUs x 16; fewer GPUs
accumulate gradients). Data: `bash download.sh t2i-data`. Runs are `store/runs/<date>_<name>/`, with a checkpoint every
5k steps, all kept (the averages need them). Helpers for the commands below:

```bash
ck()  { ls -d store/runs/*_$1/ckpt/step$(printf %07d $(($2 * 1000))).ckpt; }   # ck <run name> <step in thousands>
cks() { for ((k = $2; k <= $3; k += $4)); do ck $1 $k; done; }                  # cks <run name> <first> <last> <every>
A=store/averages; C=scripts/t2i/ckpt; mkdir -p $A
```

## SBM-XL (`t2i/sbm_t2i_xl_step1200000_ema0_9999.pt`)

| stage | starts from | data | trains | steps |
|---|---|---|---|---|
| 1 | scratch | CC12M | everything | 0 → 1,000,000 |
| 1b | trunk: 500k–1M every 25k, weights 2^(−(1M − step) / 100k); head: 1M | CC12M | the head (Muon 5e-5, AdamW 5e-6) | → 1,005,000 |
| 2 | 1,005,000 | the mix | everything | → 1,165,000 |
| 3 | trunk: 1,020k–1,165k; head: 1,050k–1,070k (uniform, every 5k) | CC12M + the mix | the head | → 1,200,000 |

```bash
python -m sbm_t2i.cli train t2i_rae_xl1dh name=xl_s1
W=$(awk 'BEGIN { for (k = 500; k <= 1000; k += 25) printf "%s%.8g", (k > 500 ? "," : ""), 2 ^ ((k - 1000) / 100) }')
python $C/soup.py $A/xl_x100.ckpt --weights=$W $(cks xl_s1 500 1000 25)
python $C/nohead.py $A/xl_s1b.ckpt $A/xl_x100.ckpt $(ck xl_s1 1000)
python $C/soup_to_resume.py $A/xl_s1b_init.ckpt --soup $A/xl_s1b.ckpt --base $(ck xl_s1 1000)
python -m sbm_t2i.cli train t2i_rae_xl1dh +stage=headonly name=xl_s1b init.from=$A/xl_s1b_init.ckpt \
    data.mix=null optim.lr=5e-5 optim.adam_lr=5e-6 train.steps_after_init=5000
python $C/soup_to_resume.py $A/xl_s2_init.ckpt --base $(ck xl_s1b 1005)
python -m sbm_t2i.cli train t2i_rae_xl1dh +stage=mix name=xl_s2 init.from=$A/xl_s2_init.ckpt
python $C/soup.py $A/xl_a1020_1165.ckpt $(cks xl_s2 1020 1165 5)
python $C/soup.py $A/xl_a1050_1070.ckpt $(cks xl_s2 1050 1070 5)
python $C/nohead.py $A/xl_s3.ckpt $A/xl_a1020_1165.ckpt $A/xl_a1050_1070.ckpt
python $C/soup_to_resume.py $A/xl_s3_init.ckpt --soup $A/xl_s3.ckpt --base $(ck xl_s2 1165)
python -m sbm_t2i.cli train t2i_rae_xl1dh +stage=headonly name=xl_s3 init.from=$A/xl_s3_init.ckpt train.steps_after_init=35000
python sample.py t2i --ckpt $(ck xl_s3 1200)      # the released model: this checkpoint's EMA-0.9999 copy
```

The original stage-1 run used the fused attention forward and no gradient clipping for its first 103k steps; the config
has the settings of the rest of the run (and of the whole SBM-B run).

## SBM-B (`t2i/sbm_t2i_b_step1165000_ema0_9999.pt`)

| stage | starts from | data | trains | steps |
|---|---|---|---|---|
| 1 | scratch | CC12M | everything | 0 → 1,000,000 |
| 2 | trunk: 850k–1M (uniform, every 25k); head: 995k | the mix | everything | → 1,160,000 |
| 3 | trunk: 1,040k–1,160k (uniform, every 5k); head: 1,160k | CC12M + the mix | the head | → 1,165,000 |

```bash
python -m sbm_t2i.cli train t2i_rae_b1dh_t15 name=b_s1
python $C/soup.py $A/b_a850_1000.ckpt $(cks b_s1 850 1000 25)
python $C/nohead.py $A/b_s2.ckpt $A/b_a850_1000.ckpt $(ck b_s1 995)
python $C/soup_to_resume.py $A/b_s2_init.ckpt --soup $A/b_s2.ckpt --base $(ck b_s1 1000)
python -m sbm_t2i.cli train t2i_rae_b1dh_t15 +stage=mix name=b_s2 init.from=$A/b_s2_init.ckpt
python $C/soup.py $A/b_a1040_1160.ckpt $(cks b_s2 1040 1160 5)
python $C/nohead.py $A/b_s3.ckpt $A/b_a1040_1160.ckpt $(ck b_s2 1160)
python $C/soup_to_resume.py $A/b_s3_init.ckpt --soup $A/b_s3.ckpt --base $(ck b_s2 1160)
python -m sbm_t2i.cli train t2i_rae_b1dh_t15 +stage=headonly name=b_s3 init.from=$A/b_s3_init.ckpt train.steps_after_init=5000
python sample.py t2i --ckpt $(ck b_s3 1165)
```

`sample.py` and `scripts/t2i/geneval_sample.py` read a training checkpoint (with the run's `run.yaml` two folders up) at
the released models' settings: omega 0.35 and label scale 1.125 for SBM-XL, 0.3 and 1.25 for SBM-B (`sbm_t2i.infer.READS`).
