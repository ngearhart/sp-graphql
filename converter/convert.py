import argparse
import subprocess
from genericpath import exists

from os import curdir, mkdir, system, remove, listdir
from os.path import join, abspath
from shutil import copy
from time import sleep
from collections import defaultdict
import json

import io
import re
from types import SimpleNamespace
import sqlite3


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,
                    help='The name of the dataset to process')
parser.add_argument('--all', action='store_true')
parser.add_argument('--spider_path', default=abspath('../spider'))
parser.add_argument('--spegql_path', default=abspath('../SPEGQL-dataset'))

training_data = defaultdict(list)


def dump_sqlite(sqlite_file, sql_output):
    with sqlite3.connect(sqlite_file) as conn, io.open(sql_output, 'w+') as p: 
        for line in conn.iterdump(): 
            p.write('%s\n' % line)


def convert_sqlite_file(old_path, new_path):
    """Convert an input SQLite schema into a MySQL compatible schema."""

    def this_line_is_useless(line):
        useless_es = [
            'BEGIN TRANSACTION',
            'COMMIT',
            'sqlite_sequence',
            'CREATE UNIQUE INDEX',
            'PRAGMA foreign_keys=OFF',
            'PRAGMA foreign_keys = OFF',
            'PRAGMA foreign_keys=ON',
            'PRAGMA foreign_keys = ON',
        ]
        for useless in useless_es:
            if re.search(useless, line):
                return True

    searching_for_end = False
    open(new_path, "a").close()  # Touch new file just in case
    with open(old_path, 'r', encoding='utf-8') as old_file:
        with open(new_path, 'w+', encoding='utf-8') as new_file:
            new_file.writelines(["SET foreign_key_checks = 0;\n"])  # Disable fkey checks
            for line in old_file.readlines():
                if this_line_is_useless(line):
                    continue

                # this line was necessary because '');
                # would be converted to \'); which isn't appropriate
                if re.match(r".*, ''\);", line):
                    line = re.sub(r"''\);", r'``);', line)

                if re.match(r'^CREATE TABLE.*', line):
                    searching_for_end = True

                m = re.search('CREATE TABLE "?(\w*)"?(.*)', line)
                if m:
                    name, sub = m.groups()
                    line = "DROP TABLE IF EXISTS %(name)s;\nCREATE TABLE IF NOT EXISTS `%(name)s`%(sub)s\n"
                    line = line % dict(name=name, sub=sub)
                else:
                    m = re.search('INSERT INTO "(\w*)"(.*)', line)
                    if m:
                        line = 'INSERT INTO %s%s\n' % m.groups()
                        line = line.replace('"', r'\"')
                        line = line.replace('"', "'")
                line = re.sub(r"([^'])'t'(.)", "\1THIS_IS_TRUE\2", line)
                line = line.replace('THIS_IS_TRUE', '1')
                line = re.sub(r"([^'])'f'(.)", "\1THIS_IS_FALSE\2", line)
                line = line.replace('THIS_IS_FALSE', '0')

                # Add auto_increment if it is not there since sqlite auto_increments ALL
                # primary keys
                if searching_for_end:
                    if re.search(r"integer(?:\s+\w+)*\s*PRIMARY KEY(?:\s+\w+)*\s*,", line):
                        line = line.replace("PRIMARY KEY", "PRIMARY KEY AUTO_INCREMENT")
                    # replace " and ' with ` because mysql doesn't like quotes in CREATE commands 
                    if line.find('DEFAULT') == -1:
                        line = line.replace(r'"', r'`').replace(r"'", r'`')
                    else:
                        parts = line.split('DEFAULT')
                        parts[0] = parts[0].replace(r'"', r'`').replace(r"'", r'`')
                        line = 'DEFAULT'.join(parts)

                # And now we convert it back (see above)
                if re.match(r".*, ``\);", line):
                    line = re.sub(r'``\);', r"'');", line)

                if searching_for_end and re.match(r'.*\);', line):
                    searching_for_end = False

                if re.match(r"CREATE INDEX", line):
                    line = re.sub('"', '`', line)

                if re.match(r"AUTOINCREMENT", line):
                    line = re.sub("AUTOINCREMENT", "AUTO_INCREMENT", line)

                # Many datasets in SPIDER use `text` as a primary key which MySQL does not support
                if re.match(r".* text.", line):
                    line = re.sub("text", "char(255)", line)

                new_file.writelines([line])
            new_file.writelines(["SET foreign_key_checks = 1;\n"])  # Re-enable fkey checks


def convert_dataset(spider_path, dataset):
    """Run the converter on a specific dataset."""
    print("Copying SQL file...")
    # Create db_dumps folder if it doesn't exist
    if not exists(join(curdir, 'db_dumps')):
        mkdir(join(curdir, 'db_dumps'))
    dump_sqlite(join(spider_path, 'database', dataset, f'{dataset}.sqlite'), join(spider_path, 'database', dataset, 'schema_2.sql'))
    convert_sqlite_file(join(spider_path, 'database', dataset, 'schema_2.sql'), join(curdir, 'db_dumps', 'schema.sql'))
    print("Launching MySQL database...")
    system("docker compose down")
    system("docker compose up -d")
    print("Waiting for MySQL to be done loading...")
    p = SimpleNamespace()
    p.stdout = ""
    while p.stdout is None or "healthy" not in p.stdout:
        p = subprocess.run("docker inspect --format='{{.State.Health}}' converter-db-1")
        print(p.stdout)
        sleep(0.5)
    system("node index.js")
    print("Copying schema file to output folder...")
    # Create folders if they do not exist
    try:
        mkdir(join(curdir, 'output'))
    except:
        pass
    try:
        mkdir(join(curdir, 'output', dataset))
    except:
        pass
    copy(join(curdir, 'schema.graphql'), join(curdir, 'output', dataset, 'schema.graphql'))
    remove(join(curdir, 'schema.graphql'))

    print("Copying training files to output folder...")
    with open(join(curdir, 'output', dataset, 'queries.json'), 'w+', encoding='utf-8') as new_file:
        new_file.writelines(json.dumps(training_data[dataset], indent=4))

    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()

    # Load training data
    with open(join(args.spegql_path, 'dataset', 'dev.json'), 'r', encoding='utf-8') as dev:
        lines = dev.readlines()
        data = json.loads(''.join(lines))
        for obj in data:
            training_data[obj["schemaId"]].append({
                "query": obj["query"],
                "question": obj["question"]
            })
    with open(join(args.spegql_path, 'dataset', 'train.json'), 'r', encoding='utf-8') as train:
        lines = train.readlines()
        data = json.loads(''.join(lines))
        for obj in data:
            training_data[obj["schemaId"]].append({
                "query": obj["query"],
                "question": obj["question"]
            })

    if args.all:
        errors = []
        for dataset in listdir(join(args.spider_path, 'database')):
            try:
                print(' - - CONVERTING {} - -'.format(dataset))
                convert_dataset(args.spider_path, dataset)
            except KeyboardInterrupt:
                print(', '.join(errors))
                exit()
            except:
                errors.append(dataset)
        print(', '.join(errors))
    else:
        if args.dataset_name is None:
            print("Missing argument: Dataset name")
            exit(1)
        convert_dataset(args.spider_path, args.dataset_name)
