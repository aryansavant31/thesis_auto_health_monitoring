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

    main.py --fault-detection --run -- <fd_model_num>
        With this option, the fault detection model will be loaded and run for inference.


topology estimation:    
    main.py --topology-estimation --train
        With this option, a topology estimation model will be trained.

    main.py --topology-estimation --run -- <tp_model_num>
        With this option, the topology estimation model will be loaded and run for inference.


fault isolation:    
    main.py --fault-isolation --train
        With this option, a fault isolation model will be trained
        (where topology and fault detection model will be loaded as per fault_isolation.config file)

    main.py --fault-isolation --run -- <fi_model_num>
        With this option, the fault isolation model will be loaded and run for inference. 
        (Depending on the fi_model_num, the approproate fd model and tp model will be also loaded alongside 
        for fi model to run)
    
"""

import argparse
from topology_estimation.main import main as topology_estimation_main
from topology_estimation.config import TopologyEstimatorConfig

# Convert the class instance to a dictionary to add tp config params in weights and biases project

# Weights and biases project init

# if choice is topology block
    # if choice is train
        # weight and biases topology block init (add the configuration of topology_estimator here)