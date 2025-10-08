# Domain Adaptation for IFCB Plankton Classification

Pipeline: data prep → dataset creation → domain shift analysis → training/adaptation → evaluation.

## Layout
- 01_data_prep/ : scanning, WoRMS mapping, vocab, filtering
- 02_data_set_visualization/ : quick visuals
- 03_dataset_creation/ : symlinked exports + reports
- 04_analyze_domain_shift/ : histograms & shift diagnostics
- 05_model_training_and_inference/ : training code, configs, runs

Generated data and large artifacts are ignored by default (see .gitignore).
