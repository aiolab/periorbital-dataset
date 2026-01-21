# Training Segmentation Networks on Periorbital Datasets.

Full paper can be found at : https://arxiv.org/abs/2409.20407


## Usage

Datasets need to be split into train and test sets prior to training. All training is controlled by the `--train` flag in the provided shell scripts. We trained using a DeepLabV3 architecture with the feature extractor pretrained on ImageNet. Model weights can be found at:
https://huggingface.co/grnahass/periorbital_segmentation/tree/main. Graphical schematic of the training schedule can be seen below. See full paper for more details.

![image](https://github.com/user-attachments/assets/1de1f733-a1e9-4923-a6cc-bcc881547edb)

The Segment Anything Model and the UNET results were not included in the original paper, but were added to this repository to avoid import errors. Additionally, it can be fun to play with SAM and other segmentation architectures... We have found that DeepLabV3 tends to perform the best at this task.

## Package

The trained model and periorbital distance prediction pipeline are available through Pip : https://pypi.org/project/periorbital-package/. For most use cases, this package should eliminate the need to retrain from scratch.

## Dataset Access

Data can be downloaded as a zip file from https://zenodo.org/records/13916845

## Cite Us

If you found this repository useful in your work, please cite us and star this repository :D

@misc{nahass2024opensourceperiorbitalsegmentationdataset,
      title={Open-Source Periorbital Segmentation Dataset for Ophthalmic Applications}, 
      author={George R. Nahass and Emma Koehler and Nicholas Tomaras and Danny Lopez and Madison Cheung and Alexander Palacios and Jefferey Peterson and Sacha Hubschman and Kelsey Green and Chad A. Purnell and Pete Setabutr and Ann Q. Tran and Darvin Yi},
      year={2024},
      eprint={2409.20407},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.20407}, 
}


