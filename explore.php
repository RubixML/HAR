<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;
use League\Csv\Writer;

const OUTPUT_FILE = 'tsne.csv';

echo '╔═══════════════════════════════════════════════════════════════╗' . "\n";
echo '║                                                               ║' . "\n";
echo '║ HAR Dataset Visualizer using t-SNE                            ║' . "\n";
echo '║                                                               ║' . "\n";
echo '╚═══════════════════════════════════════════════════════════════╝' . "\n";
echo "\n";

$xTrainReader = Reader::createFromPath(__DIR__ . '/train/X_train.csv')
    ->setDelimiter(',')->setEnclosure('"');

$yTrainReader = Reader::createFromPath(__DIR__ . '/train/y_train.csv')
    ->setDelimiter(',')->setEnclosure('"');

$dataset = Labeled::fromIterator($xTrainReader->getRecords(),
    $yTrainReader->fetchColumn(0));

$dataset = $dataset->randomize()->head(300);

$converter = new NumericStringConverter();

$dataset->apply($converter);

$embedder = new TSNE(2, 30, 12., 1000, 0.5, 0.4, 1e-6, new Euclidean());

echo 'Embedding started ...';

$start = microtime(true);

$samples = $embedder->embed($dataset);

echo ' done  in ' . (string) (microtime(true) - $start) . ' seconds.' . "\n";

$dataset = Labeled::quick($samples, $dataset->labels());

$writer = Writer::createFromPath(OUTPUT_FILE, 'w+');
$writer->insertOne(['x', 'y', 'label']);
$writer->insertAll($dataset->zip());

echo 'Embedding saved to ' . OUTPUT_FILE . '.' . "\n";