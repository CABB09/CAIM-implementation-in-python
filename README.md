# Machine Learning Task - Discretization using CAIMD and differents methods of Classification

## Project Overview

This project focuses on using the **CAIM (Class-Attribute Interdependence Maximization)** discretization algorithm and applying machine learning models such as **Naive Bayes** and **ID3** to several datasets. The goal is to explore how discretization affects the performance of these models, particularly in classification tasks involving continuous data.

The main topics covered include:

- **Discretization** of continuous attributes using the CAIM algorithm.
- Implementation of **Naive Bayes (GaussianNB)** and **ID3** classification algorithms.
- Usage of **k-fold cross-validation** and **stratified k-fold cross-validation** to evaluate the model performance on discretized datasets.

### Author

**Carlos Alexis Barrios Bello**  
Email: zS23000636@estudiantes.uv.mx  
Master's in Artificial Intelligence  
IIIA Instituto de Investigaciones en Inteligencia Artificial, Universidad Veracruzana

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Framework](#theoretical-framework)
   - [CAIM Algorithm](#caim-algorithm)
3. [Datasets](#datasets)
4. [Code Implementation](#code-implementation)
5. [Concepts from Scikit-Learn](#concepts-from-scikit-learn)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Introduction

In machine learning, extracting meaningful information from databases often requires preprocessing. One essential step in handling continuous data is **discretization**, which transforms continuous attributes into a finite number of intervals, allowing the application of algorithms that work on discrete data.

The project applies **CAIM discretization** to several datasets and evaluates the performance of **Naive Bayes** and **ID3** algorithms using cross-validation techniques.

---

## Theoretical Framework

### CAIM Algorithm

The **CAIM** algorithm aims to maximize the interdependence between class labels and attributes, generating an optimal number of intervals for discretizing continuous features. Unlike other algorithms, CAIM does not require the user to specify the number of intervals in advance.

The CAIM algorithm consists of the following steps:

1. Initialize interval boundaries and the initial discretization scheme.
2. Iteratively add new boundaries that yield the highest CAIM value until no further improvements can be made.

The process continues until the number of intervals equals the number of classes, ensuring a balance between computational cost and maximizing class-attribute interdependence.

---

## Datasets

Several well-known datasets were used for this project, including:

1. **Iris Dataset**: Contains 150 samples of iris flowers categorized into 3 species.
2. **Dry Bean Dataset**: Consists of 13,611 samples of 7 types of dry beans, extracted using computer vision techniques.
3. **Glass Identification Dataset**: Used in criminology to classify glass fragments.
4. **Letter Recognition Dataset**: Contains pixel data of 20,000 images of handwritten letters.
5. **Seeds Dataset**: Contains information on wheat grains from three different varieties.
6. **MAGIC Gamma Telescope Dataset**: Simulated data for high-energy gamma particle detection.
7. **Rice Dataset (Cammeo and Osmancik)**: Contains morphological data on 3,810 rice grains.
8. **Wine Quality Dataset**: Contains chemical properties of red and white wines.
9. **Yeast Dataset**: Contains 17 attributes related to yeast proteins.

---

## Code Implementation

The code for the **CAIM algorithm** was implemented using **Pandas** and **NumPy**. Below is a brief explanation of the key functions:

- **CAIM Calculation**: Calculates the CAIM value for a given matrix.
  
```python
def calculate_caim(quanta_matrix):
    max_values = np.max(quanta_matrix, axis=0)
    sums = np.sum(quanta_matrix, axis=0)
    caim = np.sum((max_values ** 2) / np.where(sums == 0, 1, sums)) / len(sums)
    return caim

def create_quanta_matrix(data, attribute, intervals, classes, class_label):
    quanta_matrix = np.zeros((len(classes), len(intervals) - 1))
    for idx, cl in enumerate(classes):
        class_data = data[data[class_label] == cl][attribute]
        for i in range(1, len(intervals)):
            quanta_matrix[idx, i - 1] = class_data[(class_data >= intervals[i - 1]) & (class_data < intervals[i])].count()
    return quanta_matrix
def caim_discretization(data, attribute, class_label):
    values = data[attribute].dropna().unique()
    min_value, max_value = np.min(values), np.max(values)
    intervals = [min_value, max_value]
    global_caim = 0
    while True:
        best_caim, best_interval = global_caim, None
        for boundary in boundaries:
            if boundary not in intervals:
                test_intervals = sorted(intervals + [boundary])
                quanta_matrix = create_quanta_matrix(data, attribute, test_intervals, classes, class_label)
                caim_value = calculate_caim(quanta_matrix)
                if caim_value > best_caim:
                    best_caim, best_interval = caim_value, boundary
        if best_interval:
            intervals.append(best_interval)
            global_caim = best_caim
        else:
            break
    return intervals

```

For full implementation of Naive Bayes, ID3, k-fold cross-validation, and stratified k-fold cross-validation, refer to the provided code files.

## Concepts from Scikit-Learn
### Gaussian Naive Bayes (GaussianNB)
- What is it?: A classification technique based on Bayes' Theorem with the assumption of normally distributed input features.
- How it works: It calculates the probability of each class for a given input and assigns the class with the highest probability.
- Usage: Mainly used for classification tasks with continuous features that follow a Gaussian distribution.
### K-Fold Cross Validation (KFold)
- What is it?: A technique to split the dataset into k subsets, training the model on k-1 subsets and validating it on the remaining subset. This process is repeated k times.
- Stratified K-Fold: Ensures that each fold contains approximately the same proportion of classes as the original dataset.

## Results
The results were generated using 10-fold cross-validation and stratified k-fold cross-validation. Two tables summarize the results for Naive Bayes and ID3 classifiers across various datasets, showing accuracy (ACC) and standard deviation (STD).

### Discretized Datasets (Simple CV)
Dataset	ID3 (ACC)	NB (ACC)	ID3 (STD)	NB (STD)
Iris	0.9333	0.9400	0.0843	0.0554
Dry Bean	0.5951	0.8156	0.3086	0.0102
...	...	...	...	...
### Discretized Datasets (Stratified CV)
Dataset	ID3 (ACC)	NB (ACC)	ID3 (STD)	NB (STD)
Iris	0.9800	0.9333	0.0305	0.0667
Dry Bean	0.8365	0.8157	0.0075	0.0095
...	...	...	...	...
## Conclusion
The CAIM discretization algorithm, when applied to datasets with fewer classes and clear intervals (e.g., Iris, Seeds, and Rice), significantly improved classification accuracy. Models trained using stratified cross-validation consistently outperformed those trained with simple cross-validation, especially in datasets with unbalanced class distributions.


