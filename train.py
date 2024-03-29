print('Initializing')

import argparse
from datetime import datetime
from os import curdir
import os
from os.path import join

import tensorflow
import torch
from torch.nn.parallel import DistributedDataParallel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tensorflow.keras.callbacks import TensorBoard

from constants import DEVICE
from model import T5MultiSPModel


def pre_train(system: T5MultiSPModel, callbacks=[], logger=None) -> Trainer:
    # trainer = Trainer(num_tpu_cores=8,max_epochs=1)
    # trainer = Trainer(max_epochs=1, train_percent_check=0.1)
    # checkpoint_callback = ModelCheckpoint(
    #     filepath=os.getcwd()+'/checkpoint',
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min',
    #     prefix=''
    # )
    trainer = Trainer(gpus=[0], max_epochs=1, progress_bar_refresh_rate=1, callbacks=callbacks, logger=logger)

    # # TODO: Is train percent check the same as val check interval?
    # trainer = Trainer(gpus=1, max_epochs=1,
    #                   progress_bar_refresh_rate=1, val_check_interval=0.2)
    # trainer = Trainer(gpus=1, max_epochs=3, auto_lr_find=True, progress_bar_refresh_rate=1, train_percent_check=0.2)\
    # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2)
    # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2,checkpoint_callback=checkpoint_callback)
    # trainer = Trainer(num_tpu_cores=8,max_epochs=1, progress_bar_refresh_rate=1)
    # import gc
    # gc.collect()

    trainer.fit(system)
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


def fine_tune(system: T5MultiSPModel, callbacks=[], gpus=1, logger=None) -> Trainer:
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
                      progress_bar_refresh_rate=1, val_check_interval=0.5, callbacks=callbacks, logger=logger)
    # trainer = Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=1, val_check_interval=0.5)
    # trainer = Trainer(gpus=1,max_epochs=3, progress_bar_refresh_rate=1,checkpoint_callback=checkpoint_callback)
    # trainer = Trainer(num_tpu_cores=8,max_epochs=1, progress_bar_refresh_rate=1)
    trainer.fit(system)
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
    trainer.save_checkpoint(f'fine-tuned-{datetime.now().strftime("%Y%m%d-%H%M%S")}.ckpt')
    # !zip -r finished_train.zip finished.ckpt
    # system = system.load_from_checkpoint('finished.ckpt')
    # system.task = 'finetune'
    # trainer = Trainer(gpus=gpus, max_epochs=0,
    #                   progress_bar_refresh_rate=1, val_check_interval=0.5)
    # trainer.fit(system) #, callbacks=callbacks)
    # # TODO: Running fit moves the system to CPU
    # system = system.to(DEVICE)
    return trainer


def test(system: T5MultiSPModel, trainer: Trainer, test_flag: str):
    system.num_beams = 3
    system.test_flag = test_flag
    system.prepare_data()
    trainer.test(system, dataloaders=system.test_dataloader())


def generate_system():
    hparams = argparse.Namespace(
        **{'lr': 0.0004365158322401656})  # for 3 epochs
    # system = ConvBartSystem(dataset, train_sampler, batch_size=2)

    print("Creating model...")
    # Change batch size depending on how much memory your GPU has
    system = T5MultiSPModel(hparams, batch_size=32)
    # system = T5MultiSPModel(hparams, batch_size=2)
    # system.lr = 3e-4
    
    # Set device
    system = system.to(DEVICE)
    return system


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-checkpoint', type=str, default=None)
    args = parser.parse_args()

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

    system = generate_system()

    # Tensorboard setup
    log_dir = join(curdir, "logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True, 
        write_images=True
    )

    tensorboard_logger = TensorBoardLogger(
        'tb_logs', name='spegql-run'
    )

    trainer = None
    if args.test_checkpoint is None:
        print("Pretraining...")
        pre_train(system, logger=tensorboard_logger)

        print("Fine tuning...")
        trainer = fine_tune(system, gpus=[0], logger=tensorboard_logger)
        trainer.save_checkpoint(f'post-tuning-{datetime.now().strftime("%Y%m%d-%H%M%S")}.ckpt')
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        trainer = Trainer(max_epochs=5,
                        progress_bar_refresh_rate=1, val_check_interval=0.5)
        # print('Initializing process group...')
        # torch.distributed.init_process_group(
        #     backend='nccl', world_size=8, rank=0
        # )
        # print('Setting up parallel data system')
        # system = DistributedDataParallel(system)
        # print('Done')
        system.load_from_checkpoint(args.test_checkpoint)
        system.task = 'finetune'

    print("Testing...")
    test(system, trainer, 'graphql')
    trainer.save_checkpoint(f'post-graphql-{datetime.now().strftime("%Y%m%d-%H%M%S")}.ckpt')
    test(system, trainer, 'sql')
    trainer.save_checkpoint(f'finished-{datetime.now().strftime("%Y%m%d-%H%M%S")}.ckpt')

    print(f'All done! You can now use the file finished-{datetime.now().strftime("%Y%m%d-%H%M%S")}.ckpt with server.py.')

if __name__ == "__main__":
    main()
