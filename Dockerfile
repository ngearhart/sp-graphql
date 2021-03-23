FROM python:3.9

# Copy local files
WORKDIR /sp-gql
COPY . .

# Expand datasets
RUN unzip SPEGQL-dataset.zip \
	&& unzip cosql_dataset.zip \
	&& unzip spider.zip

# Python dependencies
RUN pip install jsonschema pytorch-lightning \
	&& pip install install git+https://github.com/acarrera94/text-to-graphql-validation --log ./logs.txt

RUN git clone https://github.com/huggingface/transformers && \
	pip install ./transformers
