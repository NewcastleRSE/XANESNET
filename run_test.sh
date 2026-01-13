#!/bin/bash
# This script was converted from a main.yml GitHub workflow file.
set -e

# --- MLP Models ---
echo "--- Running: MLP (STD) XYZ -> XANES ---"
sed -i 's/gaussian: False/gaussian: True/' ./.github/workflows/inputs/in_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/gaussian: True/gaussian: False/' ./.github/workflows/inputs/in_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MLP (STD) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MLP (Kfold) XYZ -> XANES ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_mlp.yaml
sed -i 's/gaussian: False/gaussian: True/' ./.github/workflows/inputs/in_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/gaussian: True/gaussian: False/' ./.github/workflows/inputs/in_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MLP (Kfold) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_kfold_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MLP (Bootstrap) XYZ -> XANES ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_mlp.yaml
sed -i 's/gaussian: False/gaussian: True/' ./.github/workflows/inputs/in_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/gaussian: True/gaussian: False/' ./.github/workflows/inputs/in_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MLP (Bootstrap) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_bootstrap_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MLP (Ensemble) XYZ -> XANES ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MLP (Ensemble) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mlp_ensemble_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

# --- Other NN Architectures (STD) ---

echo "--- Running: CNN (STD) XYZ -> XANES ---"
sed -i 's/gaussian: False/gaussian: True/' ./.github/workflows/inputs/in_cnn.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/cnn_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/gaussian: True/gaussian: False/' ./.github/workflows/inputs/in_cnn.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: CNN (STD) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/cnn_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: LSTM (STD) XYZ -> XANES ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_lstm.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/lstm_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: LSTM (STD) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_lstm.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/lstm_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

# --- Autoencoder Models ---

echo "--- Running: AE_MLP (STD) XYZ -> XANES ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_MLP (STD) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_MLP (Kfold) XYZ -> XANES ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_ae_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_MLP (Kfold) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_kfold_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_ae_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_MLP (Bootstrap) XYZ -> XANES ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_ae_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_MLP (Bootstrap) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_bootstrap_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_ae_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_MLP (Ensemble) XYZ -> XANES ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_ae_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_mlp_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_MLP (Ensemble) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_mlp_ensemble_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_ae_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_CNN (STD) XYZ -> XANES ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_ae_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/ae_cnn_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AE_CNN (STD) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_ae_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/ae_cnn_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

# --- AEGAN Model ---

echo "--- Running: AEGAN (STD) ---"
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_std_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AEGAN (Kfold) ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_aegan.yaml
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_kfold_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_aegan.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AEGAN (Bootstrap) ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_aegan.yaml
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_bootstrap_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_aegan.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: AEGAN (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_aegan.yaml
python3 -m xanesnet.cli --mode train_all --in_file ./.github/workflows/inputs/in_aegan.yaml --save
python3 -m xanesnet.cli --mode predict_all --in_model models/aegan_mlp_ensemble_all_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_aegan.yaml
rm -rf ./.github/workflows/data/fe/processed*

# --- GNN Model ---

echo "--- Running: GNN (Std) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml

echo "--- Running: GNN (Kfold) ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_gnn.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_gnn.yaml

echo "--- Running: GNN (Bootstrap) ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_gnn.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_gnn.yaml

echo "--- Running: GNN (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_gnn.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_gnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/gnn_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_gnn.yaml
rm -rf ./.github/workflows/data/graph-set/processed*

# --- Transformer Model ---

echo "--- Running: Transformer (STD) ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_transformer.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/transformer_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml

echo "--- Running: Transformer (Kfold) ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_transformer.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_transformer.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/transformer_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_transformer.yaml

echo "--- Running: Transformer (Bootstrap) ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_transformer.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_transformer.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/transformer_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_transformer.yaml

echo "--- Running: Transformer (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_transformer.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_transformer.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/transformer_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict_gnn.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_transformer.yaml
rm -rf ./.github/workflows/data/graph-set/processed*

# --- MultiHead Models ---

echo "--- Running: MH-MLP (STD) XYZ -> XANES ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mh_mlp_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-MLP (STD) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mh_mlp_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
rm -rf ./.github/workflows/data/fe/processed*.

echo "--- Running: MH-MLP (Kfold) XYZ -> XANES ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_mh_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mh_mlp_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-MLP (Kfold) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mh_mlp_kfold_xanes_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_mh_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-MLP (Bootstrap) XYZ -> XANES ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_mh_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mh_mlp_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-MLP (Bootstrap) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mh_mlp_bootstrap_xanes_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_mh_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-MLP (Ensemble) XYZ -> XANES ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_mh_mlp.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mh_mlp_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-MLP (Ensemble) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mh_mlp.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mh_mlp_ensemble_xanes_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_mh_mlp.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-CNN (STD) XYZ -> XANES ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_mh_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/mh_cnn_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: MH-CNN (STD) XANES -> XYZ ---"
python3 -m xanesnet.cli --mode train_xanes --in_file ./.github/workflows/inputs/in_mh_cnn.yaml --save
python3 -m xanesnet.cli --mode predict_xyz --in_model models/mh_cnn_std_xanes_001 --in_file ./.github/workflows/inputs/in_predict_mh.yaml
rm -rf ./.github/workflows/data/fe/processed*

# --- SoftShell Models ---
echo "--- Running: SoftShell (STD) XYZ -> XANES ---"
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_softshell.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/softshell_std_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: SoftShell (Kfold) XYZ -> XANES ---"
sed -i 's/kfold: False/kfold: True/' ./.github/workflows/inputs/in_softshell.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_softshell.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/softshell_kfold_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/kfold: True/kfold: False/' ./.github/workflows/inputs/in_softshell.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: SoftShell (Bootstrap) XYZ -> XANES ---"
sed -i 's/bootstrap: False/bootstrap: True/' ./.github/workflows/inputs/in_softshell.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_softshell.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/softshell_bootstrap_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/bootstrap: True/bootstrap: False/' ./.github/workflows/inputs/in_softshell.yaml
rm -rf ./.github/workflows/data/fe/processed*

echo "--- Running: SoftShell (Ensemble) ---"
sed -i 's/ensemble: False/ensemble: True/' ./.github/workflows/inputs/in_softshell.yaml
python3 -m xanesnet.cli --mode train_xyz --in_file ./.github/workflows/inputs/in_softshell.yaml --save
python3 -m xanesnet.cli --mode predict_xanes --in_model models/softshell_ensemble_xyz_001 --in_file ./.github/workflows/inputs/in_predict.yaml
sed -i 's/ensemble: True/ensemble: False/' ./.github/workflows/inputs/in_softshell.yaml
rm -rf ./.github/workflows/data/graph-set/processed*

echo "--- All tasks completed successfully! ---"