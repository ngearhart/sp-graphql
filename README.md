# sp-graphql
Semantic Parsing to GraphQL - Based on Andre Carrera's work

## Host OS Setup
1. Download [cosql_dataset.zip](https://drive.google.com/uc?id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP)
2. Unzip cosql_dataset.zip, SPEGQL-dataset.zip, and spider.zip
3. Run `git clone https://github.com/huggingface/transformers && pip install ./transformers`
4. Run `pip install jsonschema pytorch-lightning pandas nltk numpy sentencepiece`
5. Run `pip install install git+https://github.com/acarrera94/text-to-graphql-validation`
6. Install [tensorflow requirements for your GPU](https://www.tensorflow.org/install/gpu)
7. Install PyTorch according to [this site](https://pytorch.org/get-started/locally/)

## Running
1. Run `python main.py`


There will be more runtime options in the future
