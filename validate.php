<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use League\Csv\Reader;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$samples = Reader::createFromPath('test/samples.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$labels = Reader::createFromPath('test/labels.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);

$dataset = Labeled::fromIterator($samples, $labels);

$estimator = PersistentModel::load(new Filesystem('har.model'));

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to report.json' . PHP_EOL;