# Patch Generation
 
This repository is based on parts of the adversarial-robustness-toolbox (ART): https://github.com/Trusted-AI/adversarial-robustness-toolbox

The repository allows to interactively generate adversarial patches for collusion attacks based on [Adversarial Patch](https://arxiv.org/pdf/1712.09665) using [Yolov5](https://github.com/ultralytics/yolov5) as a surrogate model.

## Usage:

Run the notebook [Adversarial-Patch-ART-Collusion.ipynb](Patch_Generation.ipynb), which should be self-explanatory. It will install the required dependencies automatically, so it's advised to run it in a virtual environment.

## Replication

To replicate our results, follow these steps, applied in the *Attack Parameters* section:

### Validation of our patches

Let `GENERATE = False`

For the single patches, run the notebook with `split = False` for each of the three patches, ensuring that `DISGUISE_PATH` and `PATCH_PATH` correspond to each other. An example is given for the camellia patch:
```
DISGUISE_PATH = "./disguises/camellia.jpg"
PATCH_PATH = "./trained patches/patch_camellia_v5.png"
```

For the collusion patches, run the notebook with `split = True` and:
```
DISGUISE_PATH = "./disguises/apple_split.jpg"
PATCH_PATH = "./trained patches/patch_as_apple_split_3.png"
```

In the *Results* section, the Attack Success Rate (threhsold of 50%) will be shown, as well as the CCDF curves. In the *Patch Validation* section above, the actual predictions on the patched validation images are shown.

### Generating Patches

Let `GENERATE = True` and select a `DISGUISE_PATH`. We set `USE_PRETRAINED = DISGUISE_PATH` to speed up the training. Set `split` to True for collusion attacks.

**NOTE**: To simplify backwards compatibility of the implementation, collusion patches are implemented as single patches that are split into a left and a right patch. This means the horizontal resolution of the individual collusion patches is half of their vertical resolution, as they are stretched horizontally during the application to become square. This has to be considered when applying the collusion patches to pedestrian textures, as this stretching must be done manually.

