<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Other\Loggers\Screen;
use League\Csv\Reader;
use League\Csv\Writer;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$samples = Reader::createFromPath('train/samples.csv')->getRecords();

$labels = Reader::createFromPath('train/labels.csv')->fetchColumn(0);

$dataset = Labeled::fromIterator($samples, $labels);

$estimator = new PersistentModel(
    new Pipeline([
        new NumericStringConverter(),
        new GaussianRandomProjector(110),
        new ZScaleStandardizer(),
    ], new SoftmaxClassifier(200, new Momentum(0.001))),
    new Filesystem('har.model')
);

$estimator->setLogger(new Screen('HAR'));

echo 'Training ...' . PHP_EOL;

$estimator->train($dataset);

$losses = $estimator->steps();

$writer = Writer::createFromPath('progress.csv', 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_transpose([$losses]));

echo 'Progress saved to progress.csv' . PHP_EOL;

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();
}