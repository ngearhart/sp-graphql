import glob
import itertools
import json
from functools import reduce
from os import curdir
from os.path import basename, join
from pathlib import Path

import torch
# from tqdm import tqdm
from torch.utils.data import Dataset

torch.manual_seed(0)

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)


class TextToGraphQLDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, tokenizer, type_path='train.json', block_size=102):
        'Initialization'
        super(TextToGraphQLDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        self.schema_ids = []
        root_path = join(curdir, 'SPEGQL-dataset')
        dataset_path = join(root_path, 'dataset', type_path)
        # TODO open up tables.json
        # its a list of tables
        # group by db_id
        # grab column name from column_names_original ( each column name is a list of two. and the 2nd index {1} is the column name )
        # grab table names from table_names (^ same as above )
        # concat both with the english question (table names + <c> + column names + <q> english question)
        # tokenize

        # Maybe try making making more structure
        # in the concat by using primary_keys and foreign_keys

        schemas_path = join(root_path, 'Schemas')
        # schemas = glob.glob(schemas_path + '**/' + 'schema.graphql')
        schemas = glob.glob(schemas_path + '**/' + 'simpleSchema.json')

        self.max_len = 0
        self.name_to_schema = {}
        for schema_path in schemas:
            with open(schema_path, 'r', encoding='utf-8') as s:
                data = json.load(s)

                type_field_tokens = [['<t>'] + [t['name']] + ['{'] + [
                    f['name'] for f in t['fields']] + ['}'] + ['</t>'] for t in data['types']]
                type_field_flat_tokens = reduce(
                    list.__add__, type_field_tokens)

                arguments = [a['name'] for a in data['arguments']]
                schema_tokens = type_field_flat_tokens + \
                    ['<a>'] + arguments + ['</a>']

               #  tok = tokenizer.encode_plus(schema_tokens,return_tensors='pt', max_length=704, pad_to_max_length=True)
               #  this_len = tok['input_ids'].squeeze().shape[0]

                path = Path(schema_path)
                schema_name = basename(str(path.parent))

                self.name_to_schema[schema_name] = schema_tokens

               #  self.max_name = schema_name if this_len > self.max_len else self.max_name
               #  self.max_len = this_len if this_len > self.max_len else self.max_len


        # graphql schemas
        # for schema_path in schemas:
        #   p = re.compile('\s*"""[\s\S]*?"""')
        #   pt = re.compile(': \[?\w+\!?]?!?')
        #   ps = re.compile('\s')
        #   with open(schema_path, 'r', encoding='utf-8') as s:
        #     schema = s.read()
        #     schema = p.sub('', schema)
        #     schema = pt.sub('', schema)
        #     schema = ps.sub(' ', schema)
        #     path = Path(schema_path)
        #     schema_name = basename(str(path.parent))
        #     self.name_to_schema[schema_name] = schema
        #     tok = tokenizer.batch_encode_plus([schema],return_tensors='pt')
        #     this_len = tok['input_ids'].squeeze().shape[0]
        #     self.max_name = schema_name if this_len > self.max_len else self.max_name
        #     self.max_len = this_len if this_len > self.max_len else self.max_len
        #     s.close()

        # should I be saving each schema?
        # it's more memory efficent if I only load and tokenize it once.

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for element in data:
               # db = grouped_dbs[element['db_id']]

               # tables_names = " ".join(db['table_names_original'])

               # columns_names = " ".join([column_name[1] for column_name in db['column_names_original'] ])

               # db_with_question = tables_names + ' <c> ' + columns_names + ' <q> ' + element['question']
               # max of both = 704 + 49 = 753
               # could be a little smaller
                question_with_schema = 'translate English to GraphQL: ' + \
                    element['question'] + ' ' + \
                    ' '.join(
                        self.name_to_schema[element['schemaId']]) + ' </s>'
                # print(question_with_schema)
                tokenized_s = tokenizer.encode_plus(
                    question_with_schema, max_length=1024, pad_to_max_length=True, truncation=True, return_tensors='pt')
                self.source.append(tokenized_s)
                # get max_len of source. so far it is 49
                # tokenized_s = tokenizer.batch_encode_plus([element['question']],return_tensors='pt')
                # this_len = tokenized_s['input_ids'].squeeze().shape[0]
                # self.max_len = this_len if this_len > self.max_len else self.max_len

                tokenized_t = tokenizer.encode_plus(
                    element['query'] + ' </s>', max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt')
                self.target.append(tokenized_t)
                self.schema_ids.append(element['schemaId'])

    def get_question_with_schema(self, question, schemaId):
        return 'translate English to GraphQL: ' + question + ' ' + ' '.join(self.name_to_schema[schemaId]) + ' </s>'

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

    def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]['input_ids'].squeeze()
        target_ids = self.target[index]['input_ids'].squeeze()
        src_mask = self.source[index]['attention_mask'].squeeze()

        return {
            'source_ids': source_ids,
            'source_mask': src_mask,
            'target_ids': target_ids,
            'target_ids_y': target_ids
        }


class MaskGraphQLDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, tokenizer, type_path='train.json', block_size=64):
        'Initialization'
        super(MaskGraphQLDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        path = join(curdir, 'SPEGQL-dataset', 'dataset', type_path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # for element in data:
            for example in data:
                # repeat the squence for the amount of tokens.
                # loop through those sequences and replace a different token in each one.
                # the target will be that token.
                utterance = example['query']
                # tokens = utterance.split()
                encoded_source = tokenizer.encode(
                    utterance + ' </s>', max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt').squeeze()
                token_count = encoded_source.shape[0]
                # print(encoded_source.shape)
                repeated_utterance = [
                    encoded_source for _ in range(token_count)]
                for pos in range(1, token_count):
                    encoded_source = repeated_utterance[pos].clone()
                    target_id = encoded_source[pos].item()
                    if target_id == tokenizer.eos_token_id:
                        break
                    encoded_source[pos] = tokenizer.mask_token_id
                    decoded_target = ''.join(
                        tokenizer.convert_ids_to_tokens([target_id])) + ' </s>'
                    encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4,
                                                      pad_to_max_length=True, truncation=True).squeeze()  # should always be of size 1
                    self.target.append(encoded_target)
                    self.source.append(encoded_source)

                    # repeated_utterance[pos][pos] = target_token # so that the next iteration the previous token is correct

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

    def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]  # ['input_ids'].squeeze()
        target_id = self.target[index]  # ['input_ids'].squeeze()
        # src_mask = self.source[index]['attention_mask'].squeeze()
        return {'source_ids': source_ids,
                'target_id': target_id}
        # 'source_mask': src_mask,
        # 'target_ids': target_ids,
        # 'target_ids_y': target_ids}


class SpiderDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, tokenizer, type_path='train_spider.json', block_size=102):
        'Initialization'
        super(SpiderDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        spider_path = join(curdir, 'spider')
        path = spider_path + type_path
        # TODO open up tables.json
        # its a list of tables
        # group by db_id
        # grab column name from column_names_original ( each column name is a list of two. and the 2nd index {1} is the column name )
        # grab table names from table_names (^ same as above )
        # concat both with the english question (table names + <c> + column names + <q> english question)
        # tokenize

        # Maybe try making making more structure
        # in the concat by using primary_keys and foreign_keys

        tables_path = join(spider_path, 'tables.json')

        with open(path, 'r', encoding='utf-8') as f, open(tables_path, 'r', encoding='utf-8') as t:
            databases = json.load(t)
            data = json.load(f)

            # groupby db_id
            grouped_dbs = {}
            for db in databases:
                grouped_dbs[db['db_id']] = db
            # print(grouped_dbs)
            # end grop tables

            for element in data:
                db = grouped_dbs[element['db_id']]

                # tables_names = " ".join(db['table_names_original'])
                db_tables = db['table_names_original']

                # columns_names = " ".join([column_name[1] for column_name in db['column_names_original'] ])
                tables_with_columns = ''
                for table_id, group in itertools.groupby(db['column_names_original'], lambda x: x[0]):
                    if table_id == -1:
                        continue

                    columns_names = " ".join(
                        [column_name[1] for column_name in group])
                    tables_with_columns += '<t> ' + \
                        db_tables[table_id] + ' <c> ' + \
                        columns_names + ' </c> ' + '</t> '

                # group columns with tables.

                db_with_question = 'translate English to SQL: ' + \
                    element['question'] + ' ' + tables_with_columns + '</s>'
                # question_with_schema = 'translate English to GraphQL: ' + element['question']  + ' ' + ' '.join(self.name_to_schema[element['schemaId']]) + ' </s>'

                tokenized_s = tokenizer.batch_encode_plus(
                    [db_with_question], max_length=1024, pad_to_max_length=True, truncation=True, return_tensors='pt')
                # what is the largest example size?
                # the alternative is to collate
                # might need to collate
                self.source.append(tokenized_s)

                tokenized_t = tokenizer.batch_encode_plus(
                    [element['query'] + ' </s>'], max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt')
                self.target.append(tokenized_t)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

    def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]['input_ids'].squeeze()
        target_ids = self.target[index]['input_ids'].squeeze()
        src_mask = self.source[index]['attention_mask'].squeeze()
        return {'source_ids': source_ids,
                'source_mask': src_mask,
                'target_ids': target_ids,
                'target_ids_y': target_ids}


class CoSQLMaskDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, tokenizer, type_path='cosql_train.json', block_size=64):
        'Initialization'
        super(CoSQLMaskDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        path = join(curdir, 'cosql_dataset', 'sql_state_tracking', type_path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for element in data:
                for interaction in element['interaction']:
                    # repeat the squence for the amount of tokens.
                    # loop through those sequences and replace a different token in each one.
                    # the target will be that token.
                    utterance = interaction['query']
                    # tokens = utterance.split()
                    encoded_source = tokenizer.encode(
                        utterance, max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt').squeeze()
                    token_count = encoded_source.shape[0]
                    # print(encoded_source.shape)
                    repeated_utterance = [
                        encoded_source for _ in range(token_count)]
                    for pos in range(1, token_count):
                        encoded_source = repeated_utterance[pos].clone()
                        target_id = encoded_source[pos].item()
                        if target_id == tokenizer.eos_token_id:
                            break
                        # encoded_source[pos] = tokenizer.mask_token_id
                        # self.target.append(target_id)
                        # self.source.append(encoded_source)

                        encoded_source[pos] = tokenizer.mask_token_id
                        decoded_target = ''.join(
                            tokenizer.convert_ids_to_tokens([target_id])) + ' </s>'
                        encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4,
                                                          pad_to_max_length=True, truncation=True).squeeze()  # should always be of size 1
                        self.target.append(encoded_target)
                        self.source.append(encoded_source)

                        # repeated_utterance[pos][pos] = target_token # so that the next iteration the previous token is correct

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

    def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]  # ['input_ids'].squeeze()
        target_id = self.target[index]  # ['input_ids'].squeeze()
        # src_mask = self.source[index]['attention_mask'].squeeze()
        return {'source_ids': source_ids,
                'target_id': target_id}
        # 'source_mask': src_mask,
        # 'target_ids': target_ids,
        # 'target_ids_y': target_ids}
