# Hand tracking and gesture recognition for video data
A model to track hands in a video frame and identify hand gestures as defined by HaGRID (HAnd Gesture Recognition Image Dataset). 
## Region-of-interest (ROI) filtering
The model uses image preprocessing techniques such as k-nearest-neighbours based background subtraction, adaptive thresholding and Gaussian filtering to find the area of the frame having the maximum likelihood of hands present. 

Selected ROI frames are timestamped and written to the 'handframes' folder, before being sent to the recognition model. 

## Gesture recognition
Using a ResNet-18 architecture. The approach uses transfer learning using pre-trained weights, with additional training on a subsample of the HaGRID dataset. The model achieves about 91% accuracy on pre-made annotated frames, and about 68% accuracy when tested using actual webcam footage.


The 18 gesture classes in the dataset, excluding a 19th 'No gesture' class.

![gestures](https://user-images.githubusercontent.com/65803868/185427118-2c522c62-f567-49c2-a6f2-e94a5812f052.jpg)


Confusion matrix for model when tested on annotated frames

![conf_matrix](https://user-images.githubusercontent.com/65803868/185427133-71dd07ee-02d8-4758-b0cf-9df9d4d7e0c8.png)


GIF of results

![final](https://user-images.githubusercontent.com/65803868/185431400-b7b05861-92bc-4c91-a24a-acbe9a1818c0.gif)


Predictions are written for each hand frame to the 'predictions' folder
