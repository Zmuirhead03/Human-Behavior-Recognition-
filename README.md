# # Stanford-40 Action Classifier

A simple PyTorch project that trains a ResNet-18 model on the Stanford-40 Actions dataset and classifies custom images. Includes a basic training loop, validation, model saving, and an inference script. Diagrams below show the overall workflow.

Dependencies:
pip install torch torchvision pillow tkinter matplotlib kagglehub


/project

│── Human_Behavior_Recognition.ipynb          # Training script (dataset loading, preprocessing, training, evaluation)

│── predict_single_image.py # Tkinter tool for selecting an image + showing prediction & confidence


└── stanford40_resnet18.pth  # Trained model (or download link below)

│── README.md

import kagglehub
kagglehub.dataset_download("abdullaalriad/stanford-40-full")


To Use single_image_predict.py you must have stanford40_resnet18.pth in the correct directory.

