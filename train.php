<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use League\Csv\Reader;
use League\Csv\Writer;

const MODEL_FILE = 'har.model';
const PROGRESS_FILE = 'progress.csv';
const REPORT_FILE = 'report.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ Human Activity Recognizer using Softmax Classifier            ║' . "\n";
echo '║                                                               ║' . "\n";
echo '╚═══════════════════════════════════════════════════════════════╝' . "\n";
echo "\n";

$xTrainReader = Reader::createFromPath(__DIR__ . '/train/X_train.csv')
    ->setDelimiter(',')->setEnclosure('"');

$yTrainReader = Reader::createFromPath(__DIR__ . '/train/y_train.csv')
    ->setDelimiter(',')->setEnclosure('"');

$xTestReader = Reader::createFromPath(__DIR__ . '/test/X_test.csv')
    ->setDelimiter(',')->setEnclosure('"');

$yTestReader = Reader::createFromPath(__DIR__ . '/test/y_test.csv')
    ->setDelimiter(',')->setEnclosure('"');

$training = Labeled::fromIterator($xTrainReader->getRecords(),
    $yTrainReader->fetchColumn(0));

$testing = Labeled::fromIterator($xTestReader->getRecords(),
    $yTestReader->fetchColumn(0));

$estimator = new PersistentModel(
    new Pipeline(new SoftmaxClassifier(100, new Adam(5e-4), 1e-4, 300, 1e-4, new CrossEntropy()), [
        new NumericStringConverter(),
        new SparseRandomProjector(120),
        new ZScaleStandardizer(),
    ]),
    new Filesystem(MODEL_FILE)
);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
    new PredictionSpeed(),
]);

echo 'Training started ...';

$start = microtime(true);

$estimator->train($training);

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_map(null, $estimator->steps(), []));

echo 'Progress saved to ' . PROGRESS_FILE . '.' . "\n";

echo "\n";

echo 'Generating report ...';

$start = microtime(true);

file_put_contents(REPORT_FILE, json_encode($report->generate($estimator,
    $testing), JSON_PRETTY_PRINT));

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

echo 'Report saved to ' . REPORT_FILE . '.' . "\n";

echo "\n";

$save = readline('Save this model? (y|[n]): ');

if (strtolower($save) === 'y') {
    $estimator->save();

    echo 'Model saved to ' . MODEL_FILE . '.' . "\n";
}
