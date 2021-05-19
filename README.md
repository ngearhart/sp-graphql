# sp-graphql
Semantic Parsing to GraphQL - Based on Andre Carrera's work

## Dataset Conversion
There is an automatic SQLite to GraphQL schema converter in the /converter folder.
See [converter/README.md](converter/README.md) for more information.

## Model setup
1. Download [cosql_dataset.zip](https://drive.google.com/uc?id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP)
2. Unzip cosql_dataset.zip, SPEGQL-dataset.zip, and spider.zip
3. Run `git clone https://github.com/huggingface/transformers && pip install ./transformers`
4. Run `pip install jsonschema pytorch-lightning pandas nltk numpy sentencepiece`
5. Run `pip install install git+https://github.com/acarrera94/text-to-graphql-validation`
6. Install [tensorflow requirements for your GPU](https://www.tensorflow.org/install/gpu)
7. Install PyTorch according to [this site](https://pytorch.org/get-started/locally/)

## Running
1. Run `python main.py`

## What is left to do
- See converter/README.md
- Model is still not totally working compared to Carrera's Jupyter setup
