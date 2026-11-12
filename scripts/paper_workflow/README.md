# Example Usage Workflow

The software paper's workflow configurations live in
[configs/paper_workflow/](../../configs/paper_workflow/).

The paired XYZ/spectrum dataset is available from the [NCL Research Data Repository](https://outputs.ncl.ac.uk/articles/dataset/Accurate_Affordable_and_Generalisable_Machine_Learning_Simulations_of_Transition_Metal_X-ray_Absorption_Spectra_using_the_XANESNET_Deep_Neural_Network/19087478/1). 
This workflow uses only the `totalconv` variant.

Prepare the `MultiXYZSpecSource` layout separately before running the workflows:

```bash
bash scripts/paper_workflow/prepare_workflow_data.sh \
	--source-root /path/to/XANESNET_workflow_data/datasets \
	--output-root /path/to/XANESNET_workflow_data/prepared/paper_workflow
```

Replace the `/path/to/` placeholders with local paths. The script maps each source `*_train` partition to `trainval` and each `*_test` partition to `test`. 
It creates symlinks by default; pass `--copy` to copy the files instead. The workflow configs also use placeholder `root_path` values that must point to the prepared `trainval` or `test` directory.

The `run_usage_workflows.sh` command runs training, inference, and analysis; it does not prepare the data. The training configs split `trainval` into training and validation subsets.

- [train_paper_workflow.sh](train_paper_workflow.sh)
- [infer_paper_workflow.sh](infer_paper_workflow.sh)
- [analyze_paper_workflow.sh](analyze_paper_workflow.sh)
- [run_usage_workflows.sh](run_usage_workflows.sh)

Run a stage independently after its prerequisites are available:

```bash
bash scripts/paper_workflow/train_paper_workflow.sh
bash scripts/paper_workflow/infer_paper_workflow.sh
bash scripts/paper_workflow/analyze_paper_workflow.sh
```

Each dispatcher accepts optional numbers to select specific configurations to run.
Omit the arguments to run all jobs, or provide one or more numbers to select specific jobs.

For example:
```bash
bash scripts/paper_workflow/train_paper_workflow.sh 1 # SchNet Training
bash scripts/paper_workflow/infer_paper_workflow.sh 1 # SchNet Inference
bash scripts/paper_workflow/analyze_paper_workflow.sh 2 # SchNet Analysis by elements
```

All dispatchers use `runs/paper_workflow/` by default and use the consistent model names `paper_workflow_mlp` and `paper_workflow_schnet`.
Set `OUT_DIR` to use another output location. Inference also accepts `MLP_MODEL` and `SCHNET_MODEL`;
analysis accepts `MLP_RUN` and `SCHNET_RUN` when explicit paths are needed.