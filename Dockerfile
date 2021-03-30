FROM python:3.9

# Copy local files
WORKDIR /sp-gql

# Expand datasets
COPY SPEGQL-dataset.zip SPEGQL-dataset.zip 
COPY cosql_dataset.zip cosql_dataset.zip 
COPY spider.zip spider.zip 
RUN unzip SPEGQL-dataset.zip \
	&& unzip cosql_dataset.zip \
	&& unzip spider.zip

# Python dependencies
RUN git clone https://github.com/huggingface/transformers && \
	pip install ./transformers

RUN pip install jsonschema pytorch-lightning pandas nltk numpy sentencepiece \
	&& pip install install git+https://github.com/acarrera94/text-to-graphql-validation --log ./logs.txt

COPY . .
