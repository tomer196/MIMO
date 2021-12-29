# MIMO

## Usage
### Training  
Running the training for the continuous case.
```
python train_continuous2.py --test-name 'exp_name' 
```
Running the training for the discrete case.
```
python train_discrete.py --test-name 'exp_name' 
```
 
The saved model and logs will be save in `summary/{exp_name}`.
Full list of possible arguments can be seen in `utiles.py/create_arg_parser`.  



## Repo structure
- `beamforming_test.py` - Test for understanding of the dataset and the beamforming.
- `conituous_layer.py` - Model for the continuous case.
- `selection_layer.py` - Model for the discrete case.
- `data_load.py` - PyTorch dataset & dataloader.
- `utils.py` - Helper function.

- `create_dataset` - Scripts for creation of the old (reconstruction) dataset and the new(localization, robot).  
- `Data` -  data.  
- `matlab` - Some matlab scripts used in the data acquisition.  
- `models` - Some reconstruction model we tested (currently using UNet)

## Dependencies
You can install the environment I work with using:
```
pip install -r requirment.txt
```