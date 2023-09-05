# FER-2013 Facial Expression Classification

### Author: Georgios Athanasiou

![Example Image](./images/Picture1.png) 

## Introduction

The FER-2013 dataset, composed of 35,887 grayscale images illustrating seven facial expressions, serves as a pivotal benchmark in automated emotion recognition. This repository details the approach, challenges, and results of designing a classifier for this dataset.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Installation](#installation)
3. [Strategy](#strategy)
4. [Novelty](#novelty)
5. [Results](#results)
6. [Discussion](#discussion)

## Data Preparation

The dataset revealed a skewed distribution among facial expressions. Notably, the 'disgust' class was underrepresented. Other complexities included ambiguous facial expression labels. Strategies employed included:

![Charts](./images/Picture3.png) 

- **Oversampling**: Addressing class imbalance.
- **Exclusion of 'disgust' class**: Due to its limited representation and inherent complexities.

## Installation

To set up the environment for this project:

0. The data is not available here, so modifications are required to define the paths for the data.

1. Clone this repository:

git clone <URL>

2. Install the required packages:

<pre>
```
pip install -r requirements.txt
```
</pre>   

3. **Running the Model**: Execute the main TensorFlow model using:

<pre>
```
python tf_model.py
```
</pre>

4. **Testing the Models**: There are test files located in the `tests` directory to validate the performance and functionality of the models.
5. **Exploration and Plotting**: Jupyter notebooks located in the `nb` folder provide exploratory data analysis and plotting utilities.
6. **PyTorch Files**: Alternative approaches using PyTorch can be found in the `pt_files` directory.
7. **Logs**: All logs generated during model training and evaluation are stored in the `logs` directory.
8. **Results**: The results, including model weights and evaluation metrics, can be found in the `results` directory.

## Strategy

Deep learning, specifically Convolutional Neural Networks (CNNs), was used for the classification task. The progression included:

![Architecture](./images/Picture4.png) 

- Initial CNN model with a validation accuracy of ~60%.

![Initial Results](./images/Picture5.png) 

- Benchmark research: Top Kaggle models had an average accuracy of 65%. A notable paper reached 67.2% using an ensemble approach.
- Experimentation with vgg13 and vgg16 architectures.
- Use of both TensorFlow and PyTorch for model development, with TensorFlow yielding superior results.

![Example Image](./images/Picture1.png) 

## Novelty

A two-tiered approach was employed:

1. **Binary emotion classification**: Classifying facial expressions into 'Negative' and 'Positive' emotional spectrums. Achieved a validation accuracy of 85%.
2. **Granular emotion classification**: Specific models for 'Positive' and 'Negative' emotions yielded validation accuracies of 90% and 70% respectively.

![New Approach](./images/Picture8.png) 

## Results

![Results](./images/Picture9.png) 

- **Positive expressions**: Consistent performance on both validation and test sets.
- **Negative expressions**: Improved accuracy on test set compared to existing literature models.
- **Neutral expression**: Declined performance on the test set.

![Examples](./images/Picture10.png) 

![Comparison](./images/Picture11.png) 

## Discussion

While the current progress is notable, there's room for enhancement:

- **Transfer Learning**: Greater incorporation of its benefits.
- **Exploration of architectures and loss functions**: Many avenues remain unexplored.
- **Generative Adversarial Networks (GANs)**: Using GANs for data augmentation holds potential.
- **Sophisticated Approaches**: Detecting subtle facial cues could significantly enhance the classification accuracy.

---

### Contributions

Contributions, feedback, and improvements are welcome!

