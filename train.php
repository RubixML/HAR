<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Datasets\Unlabeled;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new NDJSON('train.ndjson'));

$estimator = new PersistentModel(
    new Pipeline([
        new GaussianRandomProjector(110),
        new ZScaleStandardizer(),
    ], new SoftmaxClassifier(256, new Momentum(0.001))),
    new Filesystem('har.model')
);

$estimator->setLogger($logger);

$estimator->train($dataset);

$losses = $estimator->steps();

Unlabeled::build(array_transpose([$losses]))
    ->toCSV(['losses'])
    ->write('progress.csv');

$logger->info('Progress saved to progress.csv');

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();
}