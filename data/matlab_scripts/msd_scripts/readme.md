# Mass-Spring-Damper (MSD) system in MATLAB

The generalized_msd_2 code offers you to simulate the
dynamics of multiple masses connected via springs and dampers. The motion
is simulated in 1 dimension along the horizontal axis. 
With this toolbox, you can make various configurations of the MSD
network. The picture "msd_network_schematic" attached with these code files provides a pictorial represeantion of the various connection you can make.

![architecture](msd_network_schematic.png)

Based on the number of masses in the system, you make a machine. 
The configuration of the topology can vary between healthy and unhealthy.
There are two files to alter the system. These can be found in `.../machines/<machine_name>/<healthy or unhealthy>/<topology name>/`
`config_machine_param.m` and `generate_machine`.

- `generate_machine.m`: design the connections by adding which pairs of
masses you need the spring and damper connected. This is added in
conn.pairs varaible.
- `config_machine_param.m`: design the value of the mass, spring and
damper. There are two types of springs and dampers. (i) The ones that
connect two masses (denoted by just spring_k and damper_d) and (ii) The
ones that connect mass to a wall (wall is a immovable object with
infinite mass). These are denoted by k_wall_lin, d_wall_lin
respectively. 

#### How to run the MSD simulation
Type the commands in MATLAB comand window.

**Step 1**: Use addpath in MATLAB to add the msd_scripts directory to the MATLAB environement. 
`addpath '<your previous folders>\AFD_thesis\data\matlab_scripts\msd_scripts'`

**Step 2**: Use generate_dataset to make the MSD dataset that runs the run_dynamics and stores its results into the datasets folder. 
`generate_dataset(<machine_name>, <scenario_name>, <healthy or unhealthy>, <topology_name>, <serial_num>)`

---

**Last Updated:** January 2026
