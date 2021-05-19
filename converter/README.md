## Setup
1. Install Node.JS, Docker, and Python 3.
2. Install yarn: `npm install -g yarn`
3. Run yarn: `yarn` (make sure you are in this folder)
4. Make sure you have SPEGQL-dataset and spider unzipped and in the parent folder.


## Running
Run `python convert.py <dataset_name>` or `python convert.py --all`. <br>
You can also specify a path to Carerra's SPEGQL and spider folders (default ../SPEGQL-dataset and ../spider, respectively) by using `--spegql_path` and `--spider_path`.

## Output
The results are stored in the ./output folder. If the folder is empty, the converter failed to run.

## Notes
This mostly uses [graphql-compose-mysql](https://github.com/thejibz/graphql-compose-mysql) to perform the conversion automatically. <br>
If you get a `No such file or directory: '.\\schema.graphql'` error, that means the `index.js` converter failed to run.

## TODO
- This should probably be normalized to work with any MySQL schema, not just spider
- Make the dev/training data parsing optional
- Sometimes, the output schema fails - fix this
- Sometimes, the output schema only contains 1 GraphQL type - fix this
