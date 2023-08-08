## Aircraft Detection in very high resolution satellite images with YOLOv8

### Problem Setting
Aircraft detection from Earth observation satellite images is crucial for monitoring airport activities and mapping aircraft locations. While manual digitization is accurate, it becomes impractical for large regions or when fast assessments are required.

### Proposed Solution
In this project, we test the YOLOv8 deep learning method to detect aircraft in high-resolution satellite images from Airbus' Pleiades twin satellites. YOLOv8 is a state-of-the-art object detection algorithm that has shown promising results in various applications. Inspiration for this test stems from Jeff Faudi´s code on [Kaggle](https://www.kaggle.com/code/jeffaudi/aircraft-detection-with-yolov5).

![val_batch0_labels.jpg](images/Aircraft_predictions.png)

### Methodological Steps
1. Data Collection: We used the Airbus Intelligence dataset available on Kaggle, which contains satellite images of airports along with labeled aircraft objects.

2. Image Preprocessing: The satellite images were preprocessed to extract relevant tiles and their corresponding annotations for training the YOLOv8 model.

3. Data Augmentation: To increase the training data and improve model generalization, we applied various data augmentation techniques such as rotation, flipping, and scaling.

4. Model Calibration: We trained the YOLOv8 model on the preprocessed data to accurately detect aircraft in the satellite images.

5. Model Evaluation: After training, we evaluated the model's performance using metrics like box loss, class loss, precision, recall, and mAP.

6. Inference: The calibrated YOLOv8 model was applied to new satellite images for real-time aircraft detection.

You can add the additional information to your README.md as follows:

### Technical Details
The workflow was implemented and tested in Google Colab using a GPU runtime. This allowed for faster training and inference of the YOLOv8 model on the high-resolution satellite images.

To replicate this project and run the code successfully, the following technical requirements are recommended:

- Google Colab with GPU runtime: The GPU runtime provides significant speedup during model training and inference. It can be enabled in Google Colab under "Runtime" > "Change runtime type" > "GPU".

- Python Environment: The code was developed and tested in Python, so a working Python environment is required. The necessary packages and libraries can be installed using the provided "requirements.txt" file.

- Data Access: Ensure that the Airbus Intelligence dataset is accessible from your working directory in Google Colab. You can upload the dataset to Google Drive or directly to the Colab session.

- Model Calibration: For model calibration, the YOLOv8 model weights file ("yolov8n.pt" or "yolov8s.pt") should be available in the working directory.

- Data Preprocessing: The data preprocessing steps outlined in the code should be followed to prepare the dataset for training.

With these technical details in place, you should be able to replicate the aircraft detection workflow using YOLOv8 on Earth observation satellite images.

Please refer to the code documentation and comments for any additional information on the implementation and usage of the functions provided in this repository.

### Data Sets
The Airbus Intelligence dataset available on [Kaggle](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset) was used for this project. The dataset includes high-resolution satellite images of airports, along with corresponding annotations of aircraft objects.

For more details, you can visit the [YOLOv8 official website](https://ultralytics.com/yolov8) and the [GitHub repository](https://github.com/ultralytics/ultralytics).

*Please note that this is a summary of the steps and methods used in the project. For a detailed implementation, refer to the code and documentation provided in the repository.*
