import os


alpha = {"a0": 0.90, "a1": 0.95, "a2": 0.99}
temperature = {"t0": 1e-6,  "t1": 1e-5,  "t2": 1e-4}
start_epoch = {"se0": 2, "se1": 20}
which_blocks = {"wb0": "all", "wb1": "first", "wb2": "last"}

LR  = 1e-5
WD  = 5e-4
BS  = 8

out_dir = "."

for a_key, a_val in alpha.items():
    for t_key, t_val in temperature.items():
        for se_key, se_val in start_epoch.items():
            for wb_key, wb_val in which_blocks.items():
                filename = f"ifi_{a_key}_{t_key}_{se_key}_{wb_key}.sh"
                path = os.path.join(out_dir, filename)
                with open(path, "w") as f:
                    f.write(f"""#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name={filename.replace('.sh','')}
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source venv/bin/activate

python src/main.py \\
    --mode ifi \\
    --lr {LR} \\
    --weight_decay {WD} \\
    --batch_size {BS} \\
    --alpha {a_val} \\
    --temperature {t_val} \\
    --start_epoch {se_val} \\
    --which_blocks {wb_val} \\
    --sbatch_script {filename}
""")
