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

ini_set('memory_limit', '-1');

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Human Activity Recognizer using a Softmax Classifier          ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$trainSamples = Reader::createFromPath(__DIR__ . '/train/X_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$trainLabels = Reader::createFromPath(__DIR__ . '/train/y_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);

$testSamples = Reader::createFromPath(__DIR__ . '/test/X_test.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$testLabels = Reader::createFromPath(__DIR__ . '/test/y_test.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);

$training = Labeled::fromIterator($trainSamples, $trainLabels);

$testing = Labeled::fromIterator($testSamples, $testLabels);

$estimator = new Pipeline([
    new NumericStringConverter(),
    new SparseRandomProjector(120),
    new ZScaleStandardizer(),
], new SoftmaxClassifier(100, new Adam(0.001)));

$estimator->setLogger(new Screen('HAR'));

$estimator->train($training);

$losses = $estimator->steps();

$writer = Writer::createFromPath('progress.csv', 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_map(null, $losses, []));

echo 'Progress saved to progress.csv' . PHP_EOL;

$predictions = $estimator->predict($testing);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $testing->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to report.json' . PHP_EOL;
