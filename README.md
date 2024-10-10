# Training Segmentation Networks on Periorbital Datasets.

Full paper can be found at : https://arxiv.org/abs/2409.20407


## Usage

Datasets need to be split into train and test sets prior to training. All training is controlled by the `--train` flag in the provided shell scripts. We trained using a DeepLabV3 architecture with the feature extractor pretrained on ImageNet. Model weights can be found at:
https://huggingface.co/grnahass/periorbital_segmentation/tree/main. Graphical schematic of the training schedule can be seen below. See full paper for more details.

![image](https://github.com/user-attachments/assets/1de1f733-a1e9-4923-a6cc-bcc881547edb)


## Dataset Access

Data can be downloaded as a zip file from https://zenodo.org/records/13916845
