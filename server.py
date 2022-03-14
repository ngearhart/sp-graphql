from sys import argv
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS

from train import generate_system
app = Flask(__name__)
CORS(app)


system = generate_system()


@app.route('/', methods=['GET', 'POST'])
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    req_json = request.get_json()
    prompt = req_json['prompt']
    schemaId = req_json['schemaId']
    if system.train_dataset_g.name_to_schema[schemaId] is not None:
        input_string = system.train_dataset_g.get_question_with_schema(prompt, schemaId)
    elif system.dev_dataset.name_to_schema[schemaId] is not None:
        input_string = system.val_dataset_g.get_question_with_schema(prompt, schemaId)
    print(input_string)

    # val_inputs = system.val_dataset[0]
    # print(system.tokenizer.decode(val_inputs['source_ids'], skip_special_tokens=False))

    inputs = system.tokenizer.batch_encode_plus([input_string], max_length=1024, return_tensors='pt')['input_ids']

    print(inputs.shape)
    # print(val_inputs['source_ids'].shape)


    # generated_ids = system.model.generate(val_inputs['source_ids'].unsqueeze(0).cuda(), num_beams=1, repetition_penalty=1.0, max_length=1000, early_stopping=True)
    generated_ids = system.model.generate(inputs.cuda(), num_beams=3, repetition_penalty=1.0, max_length=1000, early_stopping=True)
    hyps = [system.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
    print(hyps)
    dict_res = { "prediction" : hyps[0]}
    print(dict_res)
    return jsonify(dict_res)


if __name__ == '__main__':
    if len(argv) < 2:
        print('Usage: python server.py <checkpoint.ckpt>')
    else:
        print(f'Loading checkpoint {argv[1]}')
        system = system.load_from_checkpoint(argv[1])
        system.task = 'finetune'
        system.prepare_data()
        app.run()
