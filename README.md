# DyDec (DynamicDeception)
DyDec (DynamicDeception) provides an implementation of a Collusion Attack on Autonomous Driving Agents.

## Structure:
* Our fork of the **adversarial-robustness-toolbox** (ART) includes the implementation of our attack algorithm
* **Patch Generation** includes the attack configuration and execution
* **Simulation Testing** includes the simulation testing in CARLA and the dataset collection


## Details

### ART

If DyDec is cloned from GitHub, to make sure the ART submodule is also cloned, run:

`git submodule update --init --recursive`

The collusion attack is implemented in [adversarial_patch_pytorch.py](adversarial-robustness-toolbox/art/attacks/evasion/adversarial_patch/adversarial_patch_pytorch.py)

### Patch Generation

To generate adversarial collusion patches or to replicate our results on the model-level, check the respective [README](<./Patch Generation/README.md>).

### Simulation Testing

To test collusion attacks on the system-level in simulation, or to collect training images, check the respective [README](<./Simulation Testing/README.md>).
