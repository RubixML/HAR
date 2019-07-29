<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\SparseRandomProjector;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;
use League\Csv\Writer;

const OUTPUT_FILE = 'embedding.csv';

ini_set('memory_limit', '-1');

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ HAR Dataset Embedder using t-SNE                              ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$samples = Reader::createFromPath(__DIR__ . '/train/X_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->getRecords();

$labels = Reader::createFromPath(__DIR__ . '/train/y_train.csv')
    ->setDelimiter(',')->setEnclosure('"')->fetchColumn(0);

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(1000);

$dataset->apply(new NumericStringConverter());
$dataset->apply(new ZScaleStandardizer());

$embedder = new TSNE(2, 30, 12.0, 10.0);

$embedder->setLogger(new Screen('HAR'));

$samples = $embedder->embed($dataset);

$writer = Writer::createFromPath(OUTPUT_FILE, 'w+');
$writer->insertOne(['x', 'y']);
$writer->insertAll($samples);

echo 'Embedding saved to ' . OUTPUT_FILE . PHP_EOL;