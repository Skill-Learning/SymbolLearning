# SymbolLearning
The main repository for the project, to learn effects and observation symbols from self-supervised robot interaction in simulation 

**:warning: Do not update the `requirements.txt` with your own local copy :warning:**

## Installation instructions 
1. Install Isaac Gym by joining the NVIDIA Developer Program [here](https://developer.nvidia.com/isaac-gym). Installations instructions are provided under `docs` in the `tarball`.
2. Install `isaacgym-utils` from [here](https://github.com/iamlab-cmu/isaacgym-utils/). The instructions are a bit outdated but should work with minor changes. Installation instructions are in the [`README`](https://github.com/iamlab-cmu/isaacgym-utils/blob/master/README.md).
3. There might be issues with installing [`dm-control`](https://pypi.org/project/dm-control/) and [`tensorflow-metadata`](https://pypi.org/project/tensorflow-metadata/).
4. `cp` one of the files, albeit for testing:

        cp franka_numerical_utils_raw.py franka_numerical_utils.py

5. Run `data_generate.py`, with `action` on [this line](https://github.com/Skill-Learning/SymbolLearning/blob/env_setup/generate_data.py#L128), set to -1. This line might change.