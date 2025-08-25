# Simulation Testing

This repository allows to test collusion attacks on the system-level in simulation.


## Installation and Usage

Download and install [PCLA](https://github.com/MasoudJTehrani/PCLA) and all of its dependencies.

Additionally, run `conda install --file requirements.txt`

### Simulation Testing of Attacks

Ensure the pedestrians in the simulation have the correct textures. Refer to this video on how to apply the textures from the `textures/` directory to pedestrians:

<p align="center">
A video tutorial on how to put the adversarial patch image on a pedestrian's shirt is available below.
  
<div align="center">
  <a href="https://youtu.be/jH6JExPmgKY"><img src="https://img.youtube.com/vi/jH6JExPmgKY/0.jpg" alt="PCLA Video Tutorial"></a>
</div>
</p>

The textures in `textures/` correspond to the single patch disguised as an apple, the apple disguise, and the collusion attack (left and right patch, both disguised as apples). `T_Male1Cloth_v1_d` is the benign texture.

Modify the file `spawnactors_validate_cli` to have the correct walker IDs for the respective textures. As this can be different on each machine (depending on which pedestrians the textures were manually applied to), we use the environment variable MACHINE_NAME to decide at runtime which IDs to use (the `texture_to_id` dictionary). The pedestrian catalogue of CARLA can be helpful in determining the IDs for the corresponding walkers that you have put the textures on: https://carla.readthedocs.io/en/latest/catalogue_pedestrians/. Additionally, set the variable PCLA_Path in `spawnactors_validate_cli` to the path to PCLA: `PCLA_PATH = "/home/path_to/PCLA"`.

Define the Scenarios in a JSON file in `scenario definitions/` and put the respective filepath in `run_all_scenarios.py` as the `SCENARIOS_FILE` variable.
`SCENARIOS` has to be `range(number of scenarios in the JSON file)`, meaning the program will iterate over it and simulate the corresponding scenarios.

To replicate our results, use:
For collusion:
```
SCENARIOS_FILE = "scenario definitions/scenarios_val_collusion_adjusted_only4.json"
SCENARIOS = list(range(14))
```
For the single patch:
```
SCENARIOS_FILE = "scenario definitions/scenarios_val_single_adjusted_only4.json"
SCENARIOS = list(range(10)) 
```

And select the Agents to use (and their seed), as well as the number of trials. Our results were obtained with `N_EXPERIMENTS = 10`, but this might take some time for all agents.

```
SEED = 0
N_EXPERIMENTS = 10
AGENTS = [ 'if_if',  f'tfpp_aim_{SEED}', f'tfpp_l6_{SEED}', f'tfpp_lav_{SEED}', f'tfpp_wp_{SEED}', "neat_neat",]
```

To finally run the simulation:

Start Unreal Engine (`make launch` in the carla directory, then click the play icon).

And run the script, providing the machine name:
`MACHINE_NAME=[name of machine] python run_all_scenarios.py`

The results (p-values, effect sizes, avg. speed, brake, ...) are saved in `plots/` as a csv file and plots respectively.

In `Results and Analysis/`, we provide the aggregated results from our experiments including the significance analysis.

### Dataset Collection

To collect a new dataset, modify `scenario definitions/dataset_scenarios.json` to contain the desired scenarios.
Also modify the `texture_to_id` mapping in `collect_dataset.py`, such that the "benign" texture is correct for the machine that is used. Run `MACHINE_NAME=[name of machine] python collect_dataset.py`. Change the DATASET_PATH variable if necessary.

To filter the collected dataset for 2 pedestrians, use `filter_dataset.py` and change the DATASET_PATH variable if necessary.


## Structure

* **run_all_scenarios.py** is the main file for running test scenarios or dataset collection. Stores the resulting statistics in the `logs` directory with the current timestamp when starting the experiment. This file will be used to do the analysis.

* **stat_analysis.py** contains the functions for the statistical analysis. It is run at the end of `run_all_scenarios.py` or manually with `Analysis_Agents.ipynb`

* **Analysis_Agents.ipynb** can be used to run `stat_analysis` manually by specifying the respective log file in the `logs` directory.

* `logs/` contains a file logging the filenames of the timeseries and images for each individual run 

* `plots/` contains the statistical analysis and plots representing each agent's behaviour averaged over the N runs

* `timeseries/` contains metrics for each individual run for every time step

* `scenario_imgs/` contains all images captured during the experiments

* `weather_presets/` contains saved weather settings for scenarios

* `scenario definitions/` contains JSON files defining the scenarios for the dataset and tests

* `Maps/` contains a selection of (customized) maps

* `textures/` contains the different pedestrian textures


**NOTE:** It is normal that the script needs multiple attempts (~10) to completely run a scenario. The script always prints the number of the current attempt. Unfortunately, it can happen that the Unreal Engine crashes, which is mostly the case when the number of failed attempts reach hundreds or thousands. Then, the Unreal Engine can be restarted and the script will continue, no need to restart the script.
