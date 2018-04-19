#!/bin/bash
echo 'Downloading data'
mkdir vec
wget https://polybox.ethz.ch/index.php/s/cpicEJeC2G4tq9U/download -O vec/pretrained_embedding.vec
wget https://polybox.ethz.ch/index.php/s/qUc2NvUh2eONfEB/download -O data.zip
mkdir data
tar -xvf data.zip -C data
rm data.zip
wget https://polybox.ethz.ch/index.php/s/HJUnOuIj3K4FEdT/download -O data/sentences.test
