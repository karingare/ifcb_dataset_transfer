# Domain Adaptation for IFCB Plankton Classification

The project is executed in six numbered steps: (1) Data prep, (2) Dataset visualization,
(3) Dataset creation, (4) Analysis of domain shift, (5) Model training and inference, and (6)
Evaluation

## Datasets used
Two publicly available datasets are used in the project, and the paths to them should be provided in 01_data_prep/roots.json.
- SYKE_plankton_IFCB_Utö_2021, 	https://doi.org/10.23728/b2share.7c273b6f409c47e98a868d6517be3ae3
- SMHI IFCB Plankton Image Reference Library https://doi.org/10.17044/scilifelab.25883455.v4


**Project Steps (1-6)**

- **01 Data Preparation**: Scan raw datasets, clean labels and build vocab. Key files: `01_data_prep/ifcb_flow.py`, `01_data_prep/run_data_selection.sh`, `01_data_prep/class_mapping.csv`, and `01_data_prep/manual_overrides.csv`. Outputs live in `01_data_prep/out_filtered_min50/`, `out_scan/` and `out_std/`.
	- Example: run the selection + filtering pipeline:
		- ``bash 01_data_prep/run_data_selection.sh``

- **02 Dataset Overlap**: Explore shared classes between datasets. Key script: `02_dataset_overlap/find_number_of_shared_classes.py`. Reports are in `02_dataset_overlap/dataset_overlap_reports/`.
	- Example: run the overlap analysis (Python env active):
		- ``python 02_dataset_overlap/find_number_of_shared_classes.py``

- **03 Dataset Creation**: Create symlinked dataset exports for training and review, plus summary reports. Key script: `03_dataset_creation/class_symlinks_and_plots.py`. Exported datasets are placed under `03_dataset_creation/exports/dataset_symlinked/`.
	- Example: create the symlinked dataset:
		- ``python 03_dataset_creation/class_symlinks_and_plots.py``

- **04 Analyze Domain Shift**: Compute feature / domain shift diagnostics to guide adaptation approach. Key script: `04_analyze_domain_shift/analyze_domain_shift.py`. Outputs go to `04_analyze_domain_shift/reports/`.
	- Example: run the domain-shift analysis:
		- ``python 04_analyze_domain_shift/analyze_domain_shift.py``

- **05 Model Training and Inference**: Train classifiers and run inference/adaptation. Main script: `05_model_training_and_inference/train_eval.py`. There is a Streamlit app to run inference with two models under `05_model_training_and_inference/app/`. Run records are under `05_model_training_and_inference/runs/`.
	- Example: bash run_all_experiments.sh
	- Example: run the web app:
		- ``cd 05_model_training_and_inference``
		- ``streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0``
		- Use SSH port forwarding from your laptop when running on a remote node: ``ssh -L 8501:<node>:8501 <user>@<host>``

- **06 Evaluation**: Collect and summarize per-class performance across shared classes and generate evaluation reports. Helper: `06_evaluation/pull_shared_class_performances.sh`. Results are in `05_model_training_and_inference/runs`
	- Example: pull performances:
		- ``bash 06_evaluation/pull_shared_class_performances.sh``

**Environments**
- This project was run in a uv environment, and requirements are specified in `envs/requirements.txt`.

## Configuration of the repo
- Generated data and large artifacts are ignored by default (see .gitignore).

## Personal notes
To run the code on dardel: 
ml tmux
tmux new -s ddls
salloc -N 1 -t 1:30:00 -A naiss2025-5-219 -p gpu 
ssh to node (unique name)
source /cfs/klemming/projects/supr/snic2020-6-126/environments/Karin/amime_uv_env/bin/activate
ml PDC
ml rocm/5.7.0
ml craype-accel-amd-gfx90a

To run the web app: execute streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
This should be done in the 05 folder. Then, from a terminal in the laptop, run "ssh -L 8501:nid002795:8501 karinga@dardel.pdc.kth.se" where 8501 is the port and nid002795 is the compute node. Open the link in browser, ex: "http://localhost:8501"



