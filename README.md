# EarlyPedestrianActionPrediction
Early prediction of pedestrian crossing action using rolling-unrolling LSTMs

## Requirments
1- Download and extract the JAAD and PIE datasets:    
   Follow the instructions provided in [https://github.com/ykotseruba/JAAD] and [https://github.com/aras62/PIE]
   
2- Setup the environment:
   ``` 
   conda env create -f env.yml
   conda activate cpred
   ```

## Training

### A. Training for Single Input Type:
1- Modify the configuration of the model to choose the input type and size in `config_file/configs_jaad.yaml` or `config_file/configs_pie.yaml`    
    
2- Run the following command for the pretraining stage:    
   `python3 train_test.py --configs_file config_file/configs_jaad.yaml --pretrain` for JAAD    
   `python3 train_test.py --configs_file config_file/configs_jaad.yaml --pretrain` for PIE (check the config file for the right configurations)    
       
3- Run the following command for the training stage:    
   `python3 train_test.py --configs_file config_file/configs_jaad.yaml` for JAAD    
   `python3 train_test.py --configs_file config_file/configs_jaad.yaml` for PIE (check the config file for the right configurations)    
       
4- Apply the training on all of the used input types    

### B. Training for the Fusion Model:
1- Modify the configuration of the model to choose the input types and sizes used in the fusion model, in `config_file/configs_fusion.yaml`    
    
2- Modify `config_file/configs_fusion.yaml` to choose the dataset    
    
3- Run the following command to train the model:    
   `python3 train_test.py --configs_file config_file/configs_fusion.yaml --fusion`

## Testing
To test the trained model, run the following command:    
`python3 train_test.py --configs_file config_file/configs_jaad.yaml --test` for single input and the JAAD dataset    
`python3 train_test.py --configs_file config_file/configs_pie.yaml --test` for single input and the PIE dataset    
`python3 train_test.py --configs_file config_file/configs_fusion.yaml --test` for the fusion model PIE/JAAD    