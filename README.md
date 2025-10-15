# Domain Adaptation for IFCB Plankton Classification

Pipeline: data prep → dataset creation → domain shift analysis → training/adaptation → evaluation.

## Layout
- 01_data_prep/ : scanning, WoRMS mapping, vocab, filtering
- 02_data_set_visualization/ : quick visuals
- 03_dataset_creation/ : symlinked exports + reports
- 04_analyze_domain_shift/ : histograms & shift diagnostics
- 05_model_training_and_inference/ : training code, configs, runs

Generated data and large artifacts are ignored by default (see .gitignore).

To run the code on dardel: 
ml tmux
tmux new -s ddls
salloc -N 1 -t 1:30:00 -A naiss2025-5-219 -p gpu 
source /cfs/klemming/projects/supr/snic2020-6-126/environments/Karin/amime_uv_env/bin/activate
ml PDC
ml rocm/5.7.0
ml craype-accel-amd-gfx90a
ssh to node (unique name).

## Datasets used
- SYKE_plankton_IFCB_Utö_2021, 	https://doi.org/10.23728/b2share.7c273b6f409c47e98a868d6517be3ae3
- SMHI IFCB Plankton Image Reference Library https://doi.org/10.17044/scilifelab.25883455.v4
