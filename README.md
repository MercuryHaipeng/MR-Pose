# MR-Pose: Towards Robust YOLO-Pose on Mosaic-Pixelated Images
This is the source code of "MR-Pose: Towards Robust YOLO-Pose on Mosaic-Pixelated Images."

Augmentor_config.py: This is the source code for the data augmentation techniques used in the paper, which employs three types of augmentation methods, with the first two randomly selected from two options.

SRD.py: This is the implementation code for the spectral energy radius random dropping method mentioned in the paper. 

Run.py: This is the model training code, which includes two components: direct model training and knowledge distillation using a teacher model. These two parts can be used separately or in combination. During training, both approaches are invoked to enhance the model's robustness against mosaic-pixelated images.
