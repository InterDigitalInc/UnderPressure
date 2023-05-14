<p align="center">
	<img src="logo.png" alt="UnderPressure's logo"/>
</p>
<h1 align="center">
	Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup
</h1>
<p align="center">
Lucas Mourot, Ludovic Hoyet, François Le Clerc and Pierre Hellier
</p>

Official implementation of our paper [Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup](https://doi.org/10.1111/cgf.14635) presented at the [Symposium on Computer Animation (SCA) 2022](http://computeranimation.org/). and published in [Computer Graphics Forum 41.8](https://onlinelibrary.wiley.com/toc/14678659/2022/41/8).

![Teaser image](/teaser.png)

Human motion synthesis and editing are essential to many applications like video games, virtual reality, and film post-production. However, they often introduce artefacts in motion capture data, which can be detrimental to the perceived realism. In particular, footskating is a frequent and disturbing artefact, which requires knowledge of foot contacts to be cleaned up. Current approaches to obtain foot contact labels rely either on unreliable threshold-based heuristics or on tedious manual annotation. In [this article](https://doi.org/10.1111/cgf.14635), we address automatic foot contact label detection from motion capture data with a deep learning based method. To this end, we first publicly release [UnderPressure](https://github.com/InterDigitalInc/UnderPressure#Database), a novel motion capture database labelled with pressure insoles data serving as reliable knowledge of foot contact with the ground. Then, we design and train a deep neural network to estimate ground reaction forces exerted on the feet from motion data and then derive accurate foot contact labels. The evaluation of our model shows that we significantly outperform heuristic approaches based on height and velocity thresholds and that our approach is much more robust when applied on motion sequences suffering from perturbations like noise or footskate. We further propose a fully automatic workflow for footskate cleanup: foot contact labels are first derived from estimated ground reaction forces. Then, footskate is removed by solving foot constraints through an optimisation-based inverse kinematics (IK) approach that ensures consistency with the estimated ground reaction forces. Beyond footskate cleanup, both the database and the method we propose could help to improve many approaches based on foot contact labels or ground reaction forces, including inverse dynamics problems like motion reconstruction and learning of deep motion models in motion synthesis or character animation.

# Database
In this work, we propose a novel database of human motion sequences captured together with pressure insoles data. For further details please refer to [our article](https://doi.org/10.1111/cgf.14635). Files from the proposed database can be downloaded from `https://files.inria.fr/UnderPressure/<subject>-<modality>.rar` where `subject` $\in \\{ S1,S2,...,S10 \\}~$ and `modality` $\in \\{$ `insoles`, `mocap-mvnx`, `mocap-mvn`, `mocap-fbx` $\\}$. `insoles` and `mocap-mvnx` files are necessary files to train or evaluate our deep neural network while `mocap-fbx` files contain motion sequences in `.fbx` format for compatibility with many softwares and `mocap-mvn` files contain raw `.mvn` motion sequences. The following table gathers links to all files:

| Subject | insoles                                                                   | mocap-mvnx                                                                      | mocap-mvn                                                                     | mocap-fbx                                                                      |
|---------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| S1      | [.../S1-insoles.rar](https://files.inria.fr/UnderPressure/S1-insoles.rar) | [.../S1-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S1-mocap-mvnx.rar) | [.../S1-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S1-mocap-mvn.rar) | [.../S1-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S1-mocap-fbx.rar)  |
| S2      | [.../S2-insoles.rar](https://files.inria.fr/UnderPressure/S2-insoles.rar) | [.../S2-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S2-mocap-mvnx.rar) | [.../S2-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S2-mocap-mvn.rar) | [.../S2-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S2-mocap-fbx.rar)  |
| S3      | [.../S3-insoles.rar](https://files.inria.fr/UnderPressure/S3-insoles.rar) | [.../S3-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S3-mocap-mvnx.rar) | [.../S3-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S3-mocap-mvn.rar) | [.../S3-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S3-mocap-fbx.rar)  |
| S4      | [.../S4-insoles.rar](https://files.inria.fr/UnderPressure/S4-insoles.rar) | [.../S4-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S4-mocap-mvnx.rar) | [.../S4-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S4-mocap-mvn.rar) | [.../S4-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S4-mocap-fbx.rar)  |
| S5      | [.../S5-insoles.rar](https://files.inria.fr/UnderPressure/S5-insoles.rar) | [.../S5-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S5-mocap-mvnx.rar) | [.../S5-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S5-mocap-mvn.rar) | [.../S5-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S5-mocap-fbx.rar)  |
| S6      | [.../S6-insoles.rar](https://files.inria.fr/UnderPressure/S6-insoles.rar) | [.../S6-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S6-mocap-mvnx.rar) | [.../S6-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S6-mocap-mvn.rar) | [.../S6-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S6-mocap-fbx.rar)  |
| S7      | [.../S7-insoles.rar](https://files.inria.fr/UnderPressure/S7-insoles.rar) | [.../S7-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S7-mocap-mvnx.rar) | [.../S7-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S7-mocap-mvn.rar) | [.../S7-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S7-mocap-fbx.rar)  |
| S8      | [.../S8-insoles.rar](https://files.inria.fr/UnderPressure/S8-insoles.rar) | [.../S8-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S8-mocap-mvnx.rar) | [.../S8-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S8-mocap-mvn.rar) | [.../S8-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S8-mocap-fbx.rar)  |
| S9      | [.../S9-insoles.rar](https://files.inria.fr/UnderPressure/S9-insoles.rar) | [.../S9-mocap-mvnx.rar](https://files.inria.fr/UnderPressure/S9-mocap-mvnx.rar) | [.../S9-mocap-mvn.rar](https://files.inria.fr/UnderPressure/S9-mocap-mvn.rar) | [.../S9-mocap-fbx.rar](https://files.inria.fr/UnderPressure/S9-mocap-fbx.rar)  |

# Implementation
We leveraged [our database](https://github.com/InterDigitalInc/UnderPressure#Database) to train a deep neural network to estimate vertical ground reaction forces (vGRFs) from motion data. We then achieve robust and accurate binary foot contact detection based on vGRFs estimation, and we propose in optimization-based inverse kinematics algorithm based on our vGRFs estimation and contact detection method to clean human motion sequence containing footskate artifacts. For further details please refer to [our article](https://doi.org/10.1111/cgf.14635). In this repository, we provide our implementation as well as a pre-trained model.

## Dependencies
* [Python 3.9.7](https://docs.python.org/3.9/)
* [Pytorch 1.10.2](https://pytorch.org/docs/1.10/)
* [[Panda3D 1.10 for visualization]](https://docs.panda3d.org/1.10/python/index)

## Get Started
To fully repoduce training and evaluations, you will need to:
1. clone the repository:
    ```
    git clone https://github.com/InterDigitalInc/UnderPressure.git
    cd UnderPressure
    ```

2. install the dependencies listed above:
    ```
	conda create -n UnderPressure python=3.9.7 -y
	conda activate UnderPressure
	conda install -c pytorch pytorch=1.10.2 cudatoolkit=11.3 -y
	pip install panda3d
    ```

3. download, extract and preprocess the database:
    ```
	cd dataset
	wget -i required_files.txt
	unrar -ap x ./*
	cd ..
	python preprocess.py
    ```

## Pre-trained Model
We provide the pre-trained model used for quantitative and qualitative evaluation in [our article](https://doi.org/10.1111/cgf.14635) packed in the archive file `pretrained.tar`. It can be loaded with the following lines of code:
```
import torch, models
checkpoint = torch.load("pretrained.tar")
model = models.DeepNetwork(state_dict=checkpoint["model"])
```
See `demo.py` for further examples on how to estimate vGRFs from motion, derive binary foot contact labels and apply our implementation of the proposed footskate cleanup approach.

## Training from Scratch
To train your own model, please run:
```
python train.py ...
```
Many arguments can be supplied e.g. hyperparameters, see `train.py` or run `python train.py -h` for further details.

### Visualization
To visualize results of vertical ground reaction forces estimation, please run:
```
python visualization.py ...
```
Few arguments can be supplied to choose the sequence to be visualized, see `visualization.py` or run `python visualization.py -h` for further details.

Footskate cleaning example visualization:
![Teaser image](/demo_footskate.gif)

## Citation
As stated in [`LICENSE.txt`](https://github.com/InterDigitalInc/UnderPressure/blob/main/LICENCE.txt), any publication resulting from the use of this work shall properly cite the latest publication of our work, which currently is:
```
@Article{
    Mourot22,
    author=        {Mourot, Lucas and Hoyet, Ludovic and Le Clerc, Fran{\c{c}}ois and Hellier, Pierre},
    title=         {UnderPressure: Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup},
    year=          2022,
    month=         dec,
    publisher=     {Wiley Online Library},
    journal=       {Computer Graphics Forum},
    volume=        {41},
    number=        {8},
    pages=         {195-206},
    numpages=      {14},
    doi=           {10.1111/cgf.14635}
}
```

## License
Copyright © 2022, InterDigital R&D France. All rights reserved.

This source code is made available under the license found in the file [`LICENSE.txt`](https://github.com/InterDigitalInc/UnderPressure/blob/main/LICENCE.txt) in the root directory of this source tree.