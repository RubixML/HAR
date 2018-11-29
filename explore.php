<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use League\Csv\Reader;
use League\Csv\Writer;

const OUTPUT_FILE = 'embedding.csv';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ HAR Dataset Embedder using PCA and t-SNE                      ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

$samples = Reader::createFromPath(__DIR__ . '/train/X_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$labels = Reader::createFromPath(__DIR__ . '/train/y_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(500);

$estimator = new Pipeline([
    new NumericStringConverter(),
    new PrincipalComponentAnalysis(100),
], new TSNE(2, 30, 12., 100., 500, 1e-8, 5, new Euclidean()));

$estimator->setLogger(new Screen('HAR'));

$estimator->train(clone $dataset); // Clone since same dataset is used later to predict

$predictions = $estimator->predict($dataset);

$writer = Writer::createFromPath(OUTPUT_FILE, 'w+');
$writer->insertOne(['x', 'y']);
$writer->insertAll($predictions);

echo 'Embedding saved to ' . OUTPUT_FILE . PHP_EOL;