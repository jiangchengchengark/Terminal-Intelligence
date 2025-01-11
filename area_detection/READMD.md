# UI Area Detection Model

This is a UI area detection model based on the YOLOv5s architecture. The model has been trained on 500 manually annotated screenshots of mobile interfaces to detect specific regions in mobile screens.

## Model Performance

After training, the model achieved the following performance metrics:

- **Training Loss**:
  - Box Loss (box_loss): **0.017337**
  - Object Loss (obj_loss): **0.055515**
  - Class Loss (cls_loss): **0.0052495**

- **Evaluation Metrics**:
  - Precision: **0.94809**
  - Recall: **0.89528**

## File Structure

```
/area_detection
│
├── test.ipynb          # Jupyter Notebook file for testing the model
├── ui_area_dev1.pt     # Trained model weights
│    
├── images/                # the directory containing the manually annotated screenshots  
|
├── output/     # the directory containing the predicted bounding boxes and scores for each image
|
├── detect.py             # script for running inference on a single image or a directory of images
|
├── detection.py              # original script for running inference on a single image or a directory of images
|    
└── README.md           # Project description file
```

## Dependencies

Please ensure the following dependencies are installed:

- Python 3.8+
- PyTorch 1.7.0+
- OpenCV
- Matplotlib
- YOLOv5


## Usage

1. **Clone the project**:

   ```bash
   git clone https://github.com/jiangchengchengark/Terminal-Intelligence.git
   cd Terminal-Intelligence/area_detection
   ```

2. **Test the model**:
   
   Open the `test.ipynb` file and run the code to test the trained model on test images.

## train detail

![results](https://github.com/user-attachments/assets/23524f1d-f67e-464e-a139-e55df53900b8)


