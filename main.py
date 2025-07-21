"""
This is the master file. 
This can run all fault detection, topology estimation and fault isolation

This will import the main files from all 3 sub modules - fault detection, isolation and topology estiation
Note: this file itself will NOT have the class code. It just imports them from the other sub modules. 


Parser arguments:
-------------------
fault detection:
    main.py --fault-detection --train
        With this option, a fault detection model will be trained. 

    main.py --fault-detection --test -- <fd_model_num>
        With this option, a fault detection model will be loaded and tested

    main.py --fault-detection --run -- <fd_model_num>
        With this option, the fault detection model will be loaded and run for inference.


topology estimation:    
    main.py --topology-estimation --train
        With this option, a topology estimation model will be trained.

    main.py --topology-estimation --test -- <tp_model_num>
        With this option, a topology estimation model will be loaded and tested.

    main.py --topology-estimation --run -- <tp_model_num>
        With this option, the topology estimation model will be loaded and run for inference.


fault isolation:    
    main.py --fault-isolation --train
        With this option, a fault isolation model will be trained
        (where topology and fault detection model will be loaded as per fault_isolation.config file)

    main.py --fault-isolation --test -- <fi_model_num>
        With this option, a fault isolation model will be loaded and tested.
        (Depending on the fi_model_num, the approproate fd model and tp model will be also loaded alongside 
        for fi model to run)

    main.py --fault-isolation --run -- <fi_model_num>
        With this option, the fault isolation model will be loaded and run for inference. 
        (Depending on the fi_model_num, the approproate fd model and tp model will be also loaded alongside 
        for fi model to run)
    
"""

import argparse
from topology_estimation.main import TrainNRIMain, PredictNRIMain

parser = argparse.ArgumentParser(description="Run the AFD implementation")
parser.add_argument('--package', type=str, 
                    choices=['fault-detection', 'topology-estimation', 'fault-isolation'],
                    required=True, help="Package to run")

parser.add_argument('--run-type', type=str, 
                    choices=['train', 'custom_test', 'predict'],
                    required=True, help="Run type: train or infer")

args = parser.parse_args()
print(f"Running '{args.run_type}' for '{args.package}' package")

if args.package == 'fault-detection':
    pass # [TODO] complete fault detection main file

if args.package == 'topology-estimation':
    if args.run_type == 'train':
        TrainNRIMain().train()
    elif args.run_type == 'custom_test':
        PredictNRIMain().test_model()
    elif args.run_type == 'predict':
        PredictNRIMain().predict()
    
    