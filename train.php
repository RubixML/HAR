<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use League\Csv\Reader;
use League\Csv\Writer;

const MODEL_FILE = 'har.model';
const PROGRESS_FILE = 'progress.csv';
const REPORT_FILE = 'report.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Human Activity Recognizer using Softmax Classifier            ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$xTrain = Reader::createFromPath(__DIR__ . '/train/X_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$yTrain = Reader::createFromPath(__DIR__ . '/train/y_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);

$xTest = Reader::createFromPath(__DIR__ . '/test/X_test.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$yTest = Reader::createFromPath(__DIR__ . '/test/y_test.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);

$training = Labeled::fromIterator($xTrain, $yTrain);
$testing = Labeled::fromIterator($xTest, $yTest);

$estimator = new Pipeline([
    new NumericStringConverter(),
    new SparseRandomProjector(120),
    new ZScaleStandardizer(),
], new SoftmaxClassifier(100, new Adam(0.002)));

$estimator->setLogger(new Screen('HAR'));

$estimator->train($training);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_map(null, $estimator->steps(), []));

echo 'Progress saved to ' . PROGRESS_FILE . PHP_EOL;

$predictions = $estimator->predict($testing);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $testing->labels());

file_put_contents(REPORT_FILE, json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to ' . REPORT_FILE . PHP_EOL;
