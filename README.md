# Arrhythmia-types-classification
This project aims to classify five types of arrhythmia including ventricular tachycardia, ventricular flutter and fibrillation, tachycardia, bradycardia, and asystole from ECG and PPG signal data.

## Dataset
The project utilizes a dataset from Saphira, T.(2022) that has already been further curated from the dataset source PhysioNet 2015 Competition resulting in a usable dataset consisting of 588 patient data containing ECG and PPG signal data with the shape of (3749,2) and sampling rate 250 Hz. full dataset description on https://archive.physionet.org/physiobank/database/challenge/2015/.

Visualization of ECG signal:
![image](https://github.com/user-attachments/assets/db5eae55-d0c4-48e8-820d-42d3e60fd658)

Visualization of PPG signal:
![image](https://github.com/user-attachments/assets/1b3c26dd-02f3-4c72-a29a-92684e04ff55)

## Methodology

We conducted a two-phase study:

1. **Baseline Model Identification**: Ablation study on neural architectures and normalization techniques.

2. **Enhanced Model Training**: Improved performance via preprocessing and targeted augmentation.

### Phase 1: Baseline Model Selection

Input pipeline: 
  - curated ECG and PPG data with the shape (588,3749,2).
  - Stratified 80-20 train-validation split.

Architecture Tested:
  - CNN
  - Hybrid CNN-LSTM

Normalization Techniques tested:
  - Z-normalization
  - Min-Max normalization

### Phase 2: Enhanced Model

Pre-processing: 
  - baseline wander removal (0.5 Hz high-pass filter).
  - Bad data elimination.

Augmentation on Imbalanced Class: 
  - +/- 1% random noise jittering
  - magnitude warping.

Normalization: 
  - min-max normalization.

Architecture: 
  - 3-block CNN


## Results and discussions

The difference between arrhythmia types when viewed from ECG and PPG signals, exists in its heart rate variability (R-R interval), P wave presence and shape, QRS complex and width, etc. Most of the characteristics lie in signal morphology and time-dependent features. Therefore, we choose CNN architecture which excels in capturing signal morphology to classify arrhythmia types by its signal morphology.

### Phase 1: Baseline Model Selection

The best performance model was achieved by a 3-block CNN with a min-max normalization technique with 93.2% accuracy.

![image](https://github.com/user-attachments/assets/91aae92f-71c6-4c76-9f65-a3363b1def9a)

We previously assumed that incorporating LSTM at the end of the architecture could add positive impact on the model performance due to its ability to capture time dependency on time-related data. However, the model shows an indication of overfitting when trained using hybrid CNN-LSTM architecture. On the other hand, a 3-block CNN architecture can generalize well on our dataset without significant overfitting. It shows that a simple CNN architecture is enough to classify arrhythmia types by looking at its signal morphology.

We also found that min-max normalization performs well on our task and dataset. This might be due to min-max normalization characteristics that preserve signal morphology similar to the raw data, only converting the range to [0,1]. On the other hand, the Z-normalization technique introduces morphology changes to the signals which can affect the classification accuracy.

### Phase 2: Enhanced Model

The enhanced model gained a 38.2% accuracy improvement.

![image](https://github.com/user-attachments/assets/32e3e70a-d4dc-4d4c-b999-6583c40ca585)

We apply bad data elimination and baseline wander removal to remove the baseline wander variation across signals. We then apply targeted data augmentation on imbalanced class data using jittering and magnitude warping to increase model performance and robustness. Hence, the model gained a 38.2% accuracy improvement on 2x more epochs than the baseline model training setup. We hypothesize that the bad data elimination and baseline wander removal contribute to the model performance improvement, while targetted data augmentation makes it more robust.


## Conclusion and Future Direction

1. A 3-block CNN architecture with a min-max normalization technique performs well on arrhythmia type classification using ECG and PPG data.
2. Applying baseline wander removal, targeted data augmentation, and bad data elimination results in a more accurate and robust classification model.

**further improvement**, apply a more generalizable data augmentation strategy to generate diverse augmented samples, effectively addressing the class imbalance problem.
