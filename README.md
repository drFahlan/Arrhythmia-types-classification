# Arrhythmia-types-classification
This project aims to classify five types of arrhythmia including ventricular tachycardia, ventricular flutter and fibrillation, tachycardia, bradycardia, and asystole using EEG and PPG signal data.

## Dataset
The project utilizes a dataset from Saphira, T. (2022) that has already been preprocessed from the dataset source PhysioNet 2015 Competition to match the needs of the task resulting in a dataset consisting of 588 patient data with each of them has ECG and PPG signal data with the shape of (3749,2).

Visualization of ECG signal:
![image](https://github.com/user-attachments/assets/db5eae55-d0c4-48e8-820d-42d3e60fd658)

Visualization of PPG signal:
![image](https://github.com/user-attachments/assets/1b3c26dd-02f3-4c72-a29a-92684e04ff55)

## Methodology
We first prepared the dataset for baseline model training by applying label encoding, followed by one-hot encoding of the labels. We investigate two normalization techniques for data normalization: Z-normalization and min-max normalization, and compare the classification performance.

Two neural network architectures were chosen to compare the performance: CNN and CNN-LSTM. *(keunggulan kedua model)

## Results 

## discussion

## Conclusion and Future Direction
