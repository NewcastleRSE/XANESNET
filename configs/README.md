# XANESNET Example Configurations

This directory holds reference YAML configurations for the three XANESNET
workflows: training, checkpointed inference, and prediction analysis. Each file
is validated by the packaged JSON Schemas under
[../xanesnet/schemas/](../xanesnet/schemas/) and may be used directly with the
`xanesnet` command-line tool.

## How configs are used

A config drives one XANESNET run.

1. **Train** — `xanesnet train -i <file>` builds the datasource, dataset,
   encodings, model, trainer, and training strategy described by the config.
   Outputs are written under `runs/<timestamp>_train_<name>/`.
2. **Infer** — `xanesnet infer -i <file> -m <checkpoint>` merges the user
   config with the checkpoint signature before validation. Predictions land in
   `runs/<timestamp>_infer_<name>/predictions/`.
3. **Analyze** — `xanesnet analyze -i <file> -r <inference-run>` computes and
   reports metrics on saved predictions.
