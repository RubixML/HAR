<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\GaussianRandomProjector;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Extractors\CSV;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$transformer = new PersistentTransformer(
    base: new Pipeline([
        new GaussianRandomProjector(112),
        new ZScaleStandardizer(),
    ]),
    persister: new Filesystem('transformer.rbx')
);

$estimator = new PersistentModel(
    base: new SoftmaxClassifier(
        batchSize: 256,
        optimizer: new Momentum(new Constant(0.001))
    ),
    persister: new Filesystem('model.rbx')
);

$dataset = Labeled::fromIterator(new NDJSON('train.ndjson'));

$dataset->apply($transformer);

[$training, $testing] = $dataset->randomize()->stratifiedSplit(0.9);

$estimator->setLogger($logger);

$estimator->setValidationDataset($testing);

$estimator->train($training);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress(), overwrite: true);

$logger->info('Progress saved to progress.csv');

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $transformer->save();
    $estimator->save();
}
