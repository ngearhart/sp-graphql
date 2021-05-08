import argparse

from os import curdir, system
from os.path import join, abspath
from shutil import copy
from time import sleep

import re

def convert_sqlite_file(old_path, new_path):

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

    def has_primary_key(line):
        return bool(re.search(r'PRIMARY KEY', line))

    searching_for_end = False
    with open(old_path, 'r', encoding='utf-8') as old_file:
        with open(new_path, 'w+', encoding='utf-8') as new_file:
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

                new_file.writelines([line])

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, nargs='+',
                    help='The name of the dataset to process')

def main(dataset):
    print("Copying SQL file...")
    convert_sqlite_file(join(abspath('..'), 'spider', 'database', dataset, 'schema.sql'), join(curdir, 'db_dumps', 'schema.sql'))
    print("Launching MySQL database...")
    system("docker compose down")
    system("docker compose up -d")
    print("Waiting for MySQL to be done loading.")
    sleep(30)  # Arbitrary, but shouldn't take more than 30 seconds for MySQL to initialize
    system("node index.js")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dataset_name[0])
