# README

## Overview
This repository has Jupyter notebooks and related assets used to train AutoML models on Azure / AWS and establish baselines locally using BHO. The baselines were established on a consumer-grade mac with 16 GB of DDR4 RAM and 8 CPUs. For the AWS training size sweep study we omitted the model artifacts because of their size (~700 mb compressed), but they can be reproduced using the jupyter script and the public aneurysm data from Emory: http://ecm2.mathcs.emory.edu/aneuriskweb/index. We used a c4.8xlarge for the main results and a c6i.8xlarge for the training size sweep.

## Contributing
Make sure to have a large enough post buffer for making commits with new notebooks if they're large:

git config --global http.postBuffer 524288000  # 500MB

Also, make sure to install git-lfs: 
```sh
brew install git-lfs
git lfs install
```

To reproduce local notebook results, use,
```sh
conda env create -f ./envs/local_env.yaml
```

