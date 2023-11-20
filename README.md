# Face Recognition Project

This project focuses on face recognition using standard face datasets, implementing face detection, and employing PCA/Eigen analysis for recognition. Additionally, it evaluates the performance and visualizes results through ROC curves. The project extends its scope by incorporating a custom dataset featuring faces from the team.

## Introduction

Face recognition is a critical aspect of computer vision and artificial intelligence. This project explores face recognition techniques using standard face datasets and extends its application to a custom dataset featuring faces from the team.

## Objective

1. **Given Standard Face Datasets**: Utilizing standard face datasets available at [http://www.face-rec.org/databases/](http://www.face-rec.org/databases/).

2. **Detecting Faces (Color or Grayscale)**: Implementing face detection techniques to identify faces within the images, considering both color and grayscale images.

3. **Recognizing Faces Based on PCA/Eigen Analysis**: Appling PCA/Eigen analysis for face recognition, leveraging dimensionality reduction techniques.

4. **Reporting Performance and Plot ROC Curve**: Evaluating the performance of the face recognition system and visualizing the results through ROC curves.

5. **Using our Own Dataset with Team Faces**: Extending the project by incorporating a custom dataset containing faces from the team members.

## Dataset

The project utilizes standard face datasets, providing a diverse set of images for training and testing the face recognition system. The datasets are obtained from [http://www.face-rec.org/databases/](http://www.face-rec.org/databases/).

## Face Detection

Face detection is implemented to identify and locate faces within images. The project supports both color and grayscale images.

## Face Recognition with PCA/Eigen Analysis

PCA/Eigen analysis is employed for face recognition, aiming to reduce the dimensionality of the feature space while preserving essential facial features.

## Custom Dataset

In addition to standard datasets, the project incorporates a custom dataset featuring faces from the team. This extends the application of face recognition to familiar faces.<br/>
The `team_images` folder contains images of team members. This custom dataset allows the project to showcase practical applications and performance on known individuals.
As we trained the model on them at the end, you can test the application using the test folder in this repo as it contains images different from the training folder

Feel free to explore and use the team dataset in your experiments.

## Streamlit Application
The model used in this application is the one that learnd our 4 members team faces
### We can choose to browse an image, use camera input or view our POC curve:
 <p align="center">
  <img align="center" src="https://github.com/alaayasser01/Face-Recognition-System/blob/main/images/choices.png"  alt="Choices">
</p>
<p align="center">
  <img align="center" src="https://github.com/alaayasser01/Face-Recognition-System/blob/main/images/Take%20a%20picture.png"  alt="Choices">
</p>

### This is a True prediction ( yeah! it's me)
 <p align="center">
  <img align="center" src="https://github.com/alaayasser01/Face-Recognition-System/blob/main/images/true%20prediction.png"  alt="True">
</p>

## ROC Curve

ROC curves are plotted to visually assess the performance of the face recognition system, depicting the trade-off between true positive rate and false positive rate.
<p align="center">
  <img align="center" src="https://github.com/alaayasser01/Face-Recognition-System/blob/main/images/output.png"  alt="ROC">
</p>

## Usage

To use the project, follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/face-recognition-project.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the face recognition script: `python face_recognition.py`

## Dependencies

Ensure you have the following dependencies installed:

- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

Install them using: `pip install -r requirements.txt`

# Your feedback and collaboration are highly valued. Feel free to delve into the details, and I'm open to discussions and contributions!
