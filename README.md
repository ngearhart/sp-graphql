# sp-graphql
Semantic Parsing to GraphQL - Based on Andre Carrera's work

## Host OS Setup
1. Run `git clone https://github.com/huggingface/transformers && pip install ./transformers`
2. Run `pip install jsonschema pytorch-lightning pandas nltk numpy sentencepiece`
3. Run `pip install install git+https://github.com/acarrera94/text-to-graphql-validation`
4. Install [tensorflow requirements for your GPU](https://www.tensorflow.org/install/gpu)
5. Install PyTorch according to [this site](https://pytorch.org/get-started/locally/)

## Docker Setup
The following assumes you have an NVidia GPU available. You must be running on a Linux distro.
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop).
2. Expose your GPU to Docker 