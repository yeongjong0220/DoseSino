# Radiation Dose-aware Sinogram Knowledge Library Transformer with Feature Modulation for Low-dose Medical Image Segmentation


# Install the environment
```
conda env create --file radon.yaml
```

## Datasets
AutoPET is available at https://autopet.grand-challenge.org/
KiTS2023 is available at https://github.com/neheller/kits23
## Training & Testing
```
# Train
python trans_train_limit.py

# Test
python trans_val_limit.py
```
