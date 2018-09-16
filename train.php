<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\Reports\AggregateReport;
use Rubix\ML\Reports\ConfusionMatrix;
use Rubix\ML\Reports\PredictionSpeed;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\Reports\MulticlassBreakdown;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use League\Csv\Reader;
use League\Csv\Writer;

const MODEL_FILE = 'har.model';
const PROGRESS_FILE = 'progress.csv';
const REPORT_FILE = 'report.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ Human Activity Recognizer using Multi Layer Neural Net        ║' . "\n";
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

$trainSamples = iterator_to_array($xTrainReader->getRecords());
$trainLabels = iterator_to_array($yTrainReader->fetchColumn(0));

$testSamples = iterator_to_array($xTestReader->getRecords());
$testLabels = iterator_to_array($yTestReader->fetchColumn(0));

$training = new Labeled($trainSamples, $trainLabels);
$testing = new Labeled($testSamples, $testLabels);

$estimator = new PersistentModel(new Pipeline(new MultiLayerPerceptron([
    new Dense(100, new He()),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.3),
    new Dense(70, new He()),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.2),
    new Dense(50, new He()),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.1),
    new Dense(30, new He()),
    new PReLU(0.2),
    new Dense(10, new He()),
    new PReLU(0.2),
], 100, new Adam(5e-4), 1e-4, new CrossEntropy(), 1e-4, new MCC(), 0.1, 3, 100), [
    new NumericStringConverter(),
    new SparseRandomProjector(200),
    new ZScaleStandardizer(),
]));

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
$writer->insertOne(['loss', 'score']);
$writer->insertAll(array_map(null, $estimator->steps(), $estimator->scores()));

echo 'Propgress saved to ' . PROGRESS_FILE . '.' . "\n";

echo "\n";

echo 'Generating report ...';

$start = microtime(true);

file_put_contents(REPORT_FILE, json_encode($report->generate($estimator,
    $testing), JSON_PRETTY_PRINT));

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

echo 'Report saved to ' . REPORT_FILE . '.' . "\n";

echo "\n";

echo 'Example predictions:' . "\n";

var_dump($estimator->proba($testing->randomize()->head(3)));

echo "\n";

$save = readline('Save this model? (y|[n]): ');

if (strtolower($save) === 'y') {
    $estimator->save(MODEL_FILE);

    echo 'Model saved to ' . MODEL_FILE . '.' . "\n";
}
