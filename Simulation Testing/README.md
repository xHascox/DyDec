# Simulation Testing

## Structure

* **run_all_scenarios.py** is the main file for running test scenarios or dataset collection. Stores the resulting statistics in the `logs` directory with the current timestamp when starting the experiment. This file will be used to do the analysis.

* **stat_analysis.py** contains the functions for the statistical analysis. It is run at the end of `run_all_scenarios.py` or manually with `Analysis_Agents.ipynb`

* **Analysis_Agents.ipynb** can be used to run `stat_analysis` manually by specifying the respective log file in the `logs` directory.

* `logs/` contains a file logging the filenames of the timeseries and images for each individual run 

* `plots/` contains the statistical analysis and plots representing each agent's behaviour averaged over the N runs

* `timeseries/` contains metrics for each individual run for every time step

* `scenario_imgs/` contains all images captured during the experiments

* `weather_presets/` contains saved weather settings for scenarios

* `scenario definitions/` contains files defining the scenarios for the dataset and tests

* `Maps/` contains a selection of (customized) maps

* `textures/` contains the different pedestrian textures


## Usage

Install CARLA Version 0.9.15

### Simulation Testing of Attacks

Ensure the pedestrians in the simulation have the correct textures, refer to this video on how to use the textures in the `textures/` directory:

<p align="center">
A video tutorial on how to put the adversarial patch image on a pedestrian's shirt is available below.
  
<div align="center">
  <a href="https://youtu.be/jH6JExPmgKY"><img src="https://img.youtube.com/vi/jH6JExPmgKY/0.jpg" alt="PCLA Video Tutorial"></a>
</div>
</p>

Modify the file `spawnactors_validate_cli` to have the correct walker IDs for the respective textures. As this can be different on each machine, we use the environment variable MACHINE_NAME to decide at runtime which IDs to use. The pedestrian catalogue of CARLA can be helpful in determining the IDs for the corresponding walkers that you have put the textures on: https://carla.readthedocs.io/en/latest/catalogue_pedestrians/. Additionally, you need to download and install [PCLA](https://github.com/MasoudJTehrani/PCLA) and set the variable PCLA_Path in `spawnactors_validate_cli` to its path: `PCLA_PATH = "/home/path_to/PCLA"`.

Define the Scenarios in a JSON file in `scenario definitions/` and put the respective filepath in `run_all_scenarios.py` as the `SCENARIOS_FILE` variable.
`SCENARIOS` has to be range(number of scenarios in the JSON file), e.g., the program will iterate over it and simulate the corresponding scenarios.

Run the simulation in Unreal Engine (`make launch` in the carla directory, then click the play icon) 

Example for running the simulation testing:
`MACHINE_NAME=vortex python run_all_scenarios.py`



### Dataset Collection

To collect a new dataset, modify `scenario definitions/dataset_scenarios.json` to contain the desired scenarios and run `collect_dataset.py`. Change the DATASET_PATH variable if necessary.

To filter the collected dataset for 2 pedestrians, use `filter_dataset.py` and change the DATASET_PATH variable if necessary.



