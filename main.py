import argparse
from datetime import datetime
from os import curdir
from os.path import join

import tensorflow
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from constants import DEVICE
from model import T5MultiSPModel


def pre_train(system: T5MultiSPModel, callbacks=[]) -> Trainer:
    # trainer = Trainer(num_tpu_cores=8,max_epochs=1)
    # trainer = Trainer(max_epochs=1, train_percent_check=0.1)
    # checkpoint_callback = ModelCheckpoint(
    #     filepath=os.getcwd()+'/checkpoint',
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min',
    #     prefix=''
    # )
    trainer = Trainer(gpus=[0], max_epochs=1, progress_bar_refresh_rate=1)

    # # TODO: Is train percent check the same as val check interval?
    # trainer = Trainer(gpus=1, max_epochs=1,
    #                   progress_bar_refresh_rate=1, val_check_interval=0.2)
    # trainer = Trainer(gpus=1, max_epochs=3, auto_lr_find=True, progress_bar_refresh_rate=1, train_percent_check=0.2)\
    # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2)
    # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2,checkpoint_callback=checkpoint_callback)
    # trainer = Trainer(num_tpu_cores=8,max_epochs=1, progress_bar_refresh_rate=1)
    # import gc
    # gc.collect()

    trainer.fit(system) #callbacks=callbacks)
    # TODO: Running fit moves the system to CPU
    system = system.to(DEVICE)
    system.tokenizer.decode(system.train_dataset[0]['source_ids'].squeeze(
    ), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    TXT = "query { faculty_aggregate { aggregate { <mask> } } } </s>"
    inputs = system.tokenizer.batch_encode_plus([TXT], return_tensors='pt')
    inputs = inputs.to(DEVICE)
    input_ids = inputs['input_ids']
    # logits = system.model(input_ids)[0]
    items = system.model.generate(input_ids.cuda())[0]
    system.tokenizer.decode(items)
    return trainer


def fine_tune(system: T5MultiSPModel, callbacks=[], gpus=1) -> Trainer:
    system.task = 'finetune'
    system.batch_size = 2  # because t5-base is smaller than bart.
    # system.lr=3e-4 # -6 is original
    # system.batch_size = 16
    system.hparams.lr = 0.0005248074602497723  # same as 5e-4
    # system.hparams.lr=3e-4
    # TODO collate to go back to 16
    # system.model.config.output_past=True
    # system.model.model.decoder.output_past=True

    system.prepare_data()  # might not be needed.

    # system.add_special_tokens()
    # system.model.output_past = True
    # trainer = Trainer(num_tpu_cores=8,max_epochs=1)
    # trainer = Trainer(max_epochs=1, train_percent_check=0.1)
    # checkpoint_callback = ModelCheckpoint(
    #     filepath=os.getcwd()+'/checkpoint_finetuning',
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min',
    #     prefix=''
    # )

    # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2)
    # trainer = Trainer(gpus=1, progress_bar_refresh_rate=1, val_check_interval=0.4)
    trainer = Trainer(gpus=gpus, max_epochs=5,
                      progress_bar_refresh_rate=1, val_check_interval=0.5)
    # trainer = Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=1, val_check_interval=0.5)
    # trainer = Trainer(gpus=1,max_epochs=3, progress_bar_refresh_rate=1,checkpoint_callback=checkpoint_callback)
    # trainer = Trainer(num_tpu_cores=8,max_epochs=1, progress_bar_refresh_rate=1)
    trainer.fit(system) #, callbacks=callbacks)
    # TODO: Running fit moves the system to CPU
    system = system.to(DEVICE)
    inputs = system.val_dataset[0]
    system.tokenizer.decode(inputs['source_ids'])
    # system.tokenizer.decode(inputs['target_ids'])

    # inputs = system.tokenizer.batch_encode_plus([user_input], max_length=1024, return_tensors='pt')
    # generated_ids = system.bart.generate(example['input_ids'].cuda(), attention_mask=example['attention_mask'].cuda(), num_beams=5, max_length=40,repetition_penalty=3.0)
    # maybe i didn't need attention_mask? or the padding was breaking something.
    # attention mask is only needed
    generated_ids = system.model.generate(inputs['source_ids'].unsqueeze(0).cuda(
    ), num_beams=5, repetition_penalty=1.0, max_length=56, early_stopping=True)
    # summary_text = system.tokenizer.decode(generated_ids[0])

    hyps = [system.tokenizer.decode(
        g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
    print(hyps)
    # trainer.save_checkpoint('finished.ckpt')
    # !zip -r finished_train.zip finished.ckpt
    system = system.load_from_checkpoint('finished.ckpt')
    system.task = 'finetune'
    trainer = Trainer(gpus=gpus, max_epochs=0,
                      progress_bar_refresh_rate=1, val_check_interval=0.5)
    trainer.fit(system) #, callbacks=callbacks)
    # TODO: Running fit moves the system to CPU
    system = system.to(DEVICE)
    return trainer


def test(system: T5MultiSPModel, trainer: Trainer, test_flag: str):
    system.num_beams = 3
    system.test_flag = test_flag
    system.prepare_data()
    trainer.test()


def main():
    num_gpus = len(tensorflow.config.list_physical_devices('GPU'))
    if num_gpus > 0:
        print("Num GPUs Available: ", num_gpus)
    else:
        print(
            "No GPU Available (tensorflow). Make sure you have tensorflow/CUDA installed.")
        return

    num_gpus_torch = torch.cuda.device_count()
    if num_gpus_torch == 0:
        print("Torch does not detect any GPUs. Try https://github.com/PyTorchLightning/pytorch-lightning/issues/1314")
        return
    else:
        print(f'{num_gpus_torch} cuda devices detected')

    hparams = argparse.Namespace(
        **{'lr': 0.0004365158322401656})  # for 3 epochs
    # system = ConvBartSystem(dataset, train_sampler, batch_size=2)

    print("Creating model...")
    # Change batch size depending on how much memory your GPU has
    # system = T5MultiSPModel(hparams, batch_size=32)
    system = T5MultiSPModel(hparams, batch_size=2)
    # system.lr = 3e-4
    
    # Set device
    system = system.to(DEVICE)

    # Tensorboard setup
    log_dir = join(curdir, "logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # print("Pretraining...")
    # pre_train(system, [tensorboard_callback])

    print("Fine tuning...")
    trainer = fine_tune(system, [tensorboard_callback], gpus=[0])
    print("Testing...")
    test(system, trainer, 'graphql')
    test(system, trainer, 'sql')

    # TODO: NTLK and Flask stuff


if __name__ == "__main__":
    main()
