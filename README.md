This repository contains a numerical solver implementing the proof of concept for the novel unified isothermal two-phase two-scale model presented in the paper ''A two-scale two-phase flow model for the separate-to-disperse phase transition in atomizing flows'' ([hal-05673253](https://hal.science/hal-05673253)).
The implementation is carried out in the framework of [samurai](https://github.com/hpc-maths/samurai).
We refer the reader to the aforementioned paper for a detailed description of the numerical strategy, which is based on an operator-splitting approach and an innovative bound-preserving strategy (see in particular Section 5 and Appendix D).

In order to build the executable, please first install ```samurai``` following the instructions at https://github.com/hpc-maths/samurai. Please, include ```nlohmann```in your ```conda```environment

```bash
conda install nlohmann_json
```

A ```paraview``` installation for the postprocessing visualization is required.

We provide a conda environment file (```conda/environment_samurai.yml```) to facilitate the setup process. You can create and activate the conda environment by running the following commands in your terminal:

```bash
conda env create -f conda/environment_samurai.yml
conda activate samurai-0.27
```

Then, move into the directory with the test case of interest and run

```bash
source configure.sh
```

Finally, to run the program, execute

```bash
./two_scale_capillarity
```

The parameters specified in ```input.json``` reproduce the air-blasted liquid column test case described in Section 5.2 of the aforementioned paper.
By default, the results are computed on a uniform mesh composed of ```512 x 256```elements, corresponding to ```8``` refinement levels in ```samurai```.
This resolution can be modified by changing the values of ```min-level``` and ```max-level```.
Finally, the inter-scale mass transfer — representing the main novelty of the proposed model — is enabled by default. It can be disabled by setting:

```bash
"mass_transfer": false
```
