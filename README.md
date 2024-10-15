# Welcome!

This is a [repository](https://github.com/AllenNeuralDynamics/lightning_pose_CO_capsule) for [lightning-pose code ocean capsule](https://codeocean.allenneuraldynamics.org/capsule/0585042/tree) showing how to train and test a LP model on the AIND behavior data step by step.
[Previous version](https://codeocean.allenneuraldynamics.org/capsule/5949595/tree) for 2023 pose-tracking Hackathon stops updating.

## Usage
- [dlc2lp_data_organization_Part0.ipynb](https://github.com/AllenNeuralDynamics/lightning_pose_CO_capsule/blob/main/code/dlc2lp_data_organization_Part0.ipynb) (created by [Shailaja](shailaja.akella@alleninstitute.org)) shows how to reorganize [DeepLabCut (DLC)](https://github.com/DeepLabCut/DeepLabCut) project structure.

- [litpose_training_Part1.ipynb](https://github.com/AllenNeuralDynamics/lightning_pose_CO_capsule/blob/main/code/litpose_training_Part1.ipynb) (created by [Di](di.wang@alleninstitute.org)) shows how to convert one DLC project to [Lightning Pose (LP)](https://github.com/danbider/lightning-pose) fromat and train a LP model step by step.

- [litpose_training_Part2.ipynb](https://github.com/AllenNeuralDynamics/lightning_pose_CO_capsule/blob/main/code/litpose_training_Part2.ipynb) (created by [Di](di.wang@alleninstitute.org)) shows how to load a pretrained LP model and perform prediction and evalutaion on the testing data.


## Materials for Lightning Pose
- [Paper](https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1) shows a detailed mathematical description of the Lightning Pose algorithm.

- [GitHub](https://github.com/danbider/lightning-pose) and [Documentation](https://lightning-pose.readthedocs.io/en/latest/index.html) show how to implement Lightning Pose.

- Reference for notebooks from Lightning Pose at [here](https://github.com/danbider/lightning-pose/blob/7da5b5e701cb315ffd6d3ac8847191ee6715c46e/scripts/litpose_training_demo.ipynb).