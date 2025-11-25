# # Stanford-40 Action Classifier

A simple PyTorch project that trains a ResNet-18 model on the Stanford-40 Actions dataset and classifies custom images. Includes a basic training loop, validation, model saving, and an inference script. Diagrams below show the overall workflow.

## Training Flow

```mermaid
flowchart TD
    A[Download Dataset] --> B[Create DataLoaders]
    B --> C[Train ResNet-18]
    C --> D[Validate Model]
    D --> E[Save .pth File]

flowchart TD
    A[Load Saved Model] --> B[Load Image Folder]
    B --> C[Transform Images]
    C --> D[Run Model]
    D --> E[Print Predicted Class]
