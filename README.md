# Rubix ML - Human Activity Recognizer
This example project demonstrates the problem of human activity recognition (HAR) using mobile phone sensor data recorded from the gyroscope and accelerometer. The training data are the human annotated sensor readings of 30 volunteers while performing various tasks such as sitting, standing, walking, and laying down. Each sample contains a window of 561 features, however, we demonstrate that with a technique called *random projection* we can reduce the dimensionality down to 110 without any loss in accuracy. The learner we'll train to accomplish our task is a [Softmax Classifier](https://docs.rubixml.com/en/latest/classifiers/softmax-classifier.html) which is the multiclass generalization of Logistic Regression.

- **Difficulty**: Medium
- **Training time**: Minutes
- **Memory needed**: 1G

## Installation
Clone the repository locally using [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/HAR
```

Install dependencies using [Composer](https://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial

### Introduction
The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (walking, walking up stairs, walking down stairs, sitting, standing, and laying) wearing a smartphone on the waist. Using its embedded accelerometer and gyroscope, 3-axial linear acceleration and 3-axial angular velocity were recorded at a constant rate of 50Hz. The sensor signals were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 seconds. Our objective is to build a classifier to recognize which of the activities a user is performing given this sensor data.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/HAR/blob/master/train.php) file in project root.

### Extracting the Data
The data are given to us in multiple CSV files within the `train` and `test` folders. Each folder contains a `samples.csv` and `labels.csv` file. Let us start by importing The PHP League's [CSV Reader](https://csv.thephpleague.com/) to help us extract the data from the source files.

```php
use League\Csv\Reader;

$samples = Reader::createFromPath('train/samples.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$labels = Reader::createFromPath('train/labels.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);
```

The `getRecords()` and `fetchColumn` methods on the Reader instance both return iterators which we'll use to instantiate a new [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object using the static `fromIterator()` method.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($samples, $labels);
```

### Dataset Preparation
The first thing we'll do to prepare the dataset is convert all numerical strings to their integer and floating point counterparts. This step is necessary because the CSV Reader imports everything as a string by default. The [Numeric String Converter](https://docs.rubixml.com/en/latest/transformers/numeric-string-converter.html) is a transformer that will handle the conversion for us.

In machine learning, dimensionality reduction is often employed to compress the input samples such that most or all of the information is preserved. By reducing the number of input features to a denser representation, we can speed up the training process by using more informative samples. [Random Projection](https://en.wikipedia.org/wiki/Random_projection) is a computationally efficient unsupervised dimensionality reduction technique based on the [Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) which states that a set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved. The next transformation we'll apply to the dataset is a [Gaussian Random Projector](https://docs.rubixml.com/en/latest/transformers/gaussian-random-projector.html) that applies a randomized linear transformation sampled from a Gaussian distribution. We'll set the target number of dimensions to 110 which is less than 20% of the original input dimensionality.

Lastly, we'll center and scale the dataset using [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html) such that the values of the features have mean 0 and unit variance. This last step will help the learner converge to a minima of the gradient quicker during training.

Since we'll be needing these transformations again in our validation script, we'll wrap them in a [Pipeline](https://docs.rubixml.com/en/latest/pipeline.html) object so that they can be persisted with the model. Pipeline is a meta-Estimator that automates the fitting and transformations of incoming datasets.

### Instantiating the Learner
The [Softmax Classifier](https://docs.rubixml.com/en/latest/classifiers/softmax-classifier.html) is a single layer neural network with a [Softmax](https://docs.rubixml.com/en/latest/neural-network/activation-functions/softmax.html) output layer. Training is done iteratively with mini batch Gradient Descent where at each epoch the model parameters take a step in the direction of the gradient of a user-defined cost function such as [Cross Entropy](https://docs.rubixml.com/en/latest/neural-network/cost-functions/cross-entropy.html).

The first hyper-parameter of Softmax Classifier is the batch size which controls the number of samples that are feed into the network at a time. The batch size trades off training speed for smoothness of the gradient estimate. A batch size of 200 works pretty well for this example.

The next hyper-parameter is the Gradient Descent optimizer. The [Momentum](https://docs.rubixml.com/en/latest/neural-network/optimizers/momentum.html) optimizer is an adaptive optimizer that adds a momentum force to every parameter update. Momentum helps to speed up the learning process by traversing the gradient quicker. It uses a global *learning rate* that can be set by the user and typically ranges from 0.1 to 0.0001. The default setting of 0.001 works fairly well for this example so we'll leave it at that.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\Persisters\Filesystem;


$estimator = new PersistentModel(
    new Pipeline([
        new NumericStringConverter(),
        new GaussianRandomProjector(110),
        new ZScaleStandardizer(),
    ], new SoftmaxClassifier(200, new Momentum(0.001))),
    new Filesystem('har.model')
);
```

We'll wrap the entire pipeline in a [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) meta-estimator that adds a `save()` and `load()` method to the base estimator. Persistent Model requires a [Persister](https://docs.rubixml.com/en/latest/persisters/api.html) object to tell it where to store the serialized model data. The [Filesystem](https://docs.rubixml.com/en/latest/persisters/filesystem.html) persister saves and loads the model data to a file located at a user-specified path in storage.

### Setting a Logger
Softmax Classifier implements the [Verbose](https://docs.rubixml.com/en/latest/verbose.html) interface and therefore can log the progress of the learner during training. To set a logger, pass a [PSR-3](https://www.php-fig.org/psr/psr-3/) compatible logger instance to the `setLogger()` method on the estimator instance. The built-in [Screen](https://docs.rubixml.com/en/latest/other/loggers/screen.html) logger is a good default choice.

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('HAR'));
```

### Training
To start training the learner, call the `train()` method on the instance with the training dataset as an input argument.

```php
$estimator->train($dataset);
```

### Training Loss
During training, the learner will record the training loss at each epoch which we can plot to visualize the training progress. To return an array with the values of the cost function at each epoch call the `steps()` method on the learner.

```php
$losses = $estimator->steps();
```

This is an example of a line plot of the Cross Entropy cost function from start to finish. You should see fast learning in the early epochs with slower learning near the end as the learner fine-tunes the model parameters.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/HAR/master/docs/images/training-loss.svg?sanitize=true)

### Saving
To save the model simply call the `save()` method on the estimator instance.

```php
$estimator->save();
```

### Cross Validation
The authors of the dataset provide an additional 2,947 labeled testing samples that we'll use to cross validate the model. We've held out using these samples for training because we'd like to be able to test the model on samples it has never seen before. We'll start by extracting the testing samples and labels from their source files.

> **Note:** The source code for this example can be found in the [validate.php](https://github.com/RubixML/HAR/blob/master/validate.php) file in project root.

```php
use League\Csv\Reader;

$samples = Reader::createFromPath('test/samples.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$labels = Reader::createFromPath('test/labels.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);
```

Then, we'll instantiate a new Labeled dataset containing the testing data.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($samples, $labels);
```

### Load Model from Storage
To load the trained Softmax Classifier pipeline, call the static `load()` method on the [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) class with a Persister instance pointing to the model in storage.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('har.model'));
```

### Making Predictions
To obtain the predictions from the model, simply pass the testing set to the `predict()` method on the estimator instance.

```php
$predictions = $estimator->predict($dataset);
```

### Generating the Report
A cross validation report gives detailed statistics about the performance of the model given the ground truth labels. A [Multiclass Breakdown](https://docs.rubixml.com/en/latest/cross-validation/reports/multiclass-breakdown.html) report breaks down the perforance of the model at the class level and outputs statistics such as accuracy, precision, recall, and more. A [Confusion Matrix](https://docs.rubixml.com/en/latest/cross-validation/reports/confusion-matrix.html) is a table that compares the predicted labels to their actual labels. We'll wrap both reports in an [Aggregate Report](https://docs.rubixml.com/en/latest/cross-validation/reports/aggregate-report.html) that combines the output of both reports into one.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);
```

Then, generate the report using the predictions and labels from the testing set.

```php
$results = $report->generate($predictions, $dataset->labels());
```

The output of the report should start with something like the output below. As you can see, our estimator is about 97% accurate and has very good specificity and negative predictive value.

```json
[
    {
        "overall": {
            "accuracy": 0.9674308943546821,
            "precision": 0.9063809316861989,
            "recall": 0.9048187793615003,
            "specificity": 0.9802554195397294,
            "negative_predictive_value": 0.9803712249716344,
            "false_discovery_rate": 0.09361906831380108,
            "miss_rate": 0.09518122063849947,
            "fall_out": 0.019744580460270538,
            "false_omission_rate": 0.01962877502836563,
            "f1_score": 0.905257137386163,
            "mcc": 0.8858111380161123,
            "informedness": 0.8850741989012301,
            "markedness": 0.8867521566578332,
            "true_positives": 2675,
            "true_negatives": 13375,
            "false_positives": 272,
            "false_negatives": 272,
            "cardinality": 2947
        },
    }

]
```

### Wrapup
- Dimensionality reduction is often used to reduce the size of the input feature space to speed up learning.
- Random Projection is a type of unsupervised dimensionality reduction based on the [Johnson–Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
- The [Softmax Classifier](https://docs.rubixml.com/en/latest/classifiers/softmax-classifier.html) is a type of single layer neural network with a Softmax output layer.
- A momentum force can be applied to the parameter updates of the network to speed up training.
- Cross Validation is the process of testing the trained estimator on data it has never seen before.

### Next Steps
Now that you've completed this tutorial on classifying human activity, see if you can acheive better accuracy by fine-tuning the hyperparameters. See how much the random projection effects the accuracy of the estimator by removing Gaussian Random Projector from the pipeline. 

## Original Dataset
Contact: Jorge L. Reyes-Ortiz(1,2), Davide Anguita(1), Alessandro Ghio(1), Luca Oneto(1) and Xavier Parra(2) Institutions: 1 - Smartlab - Non-Linear Complex Systems Laboratory DITEN - University  degli Studi di Genova, Genoa (I-16145), Italy. 2 - CETpD - Technical Research Centre for Dependency Care and Autonomous Living Polytechnic University of Catalonia (BarcelonaTech). Vilanova i la Geltrú (08800), Spain activityrecognition '@' smartlab.ws

## References:
>- Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
