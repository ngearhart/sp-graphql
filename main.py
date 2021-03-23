import argparse
from model import T5MultiSPModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

hparams = argparse.Namespace(**{'lr': 0.0004365158322401656}) # for 3 epochs
# system = ConvBartSystem(dataset, train_sampler, batch_size=2)
system = T5MultiSPModel(hparams,batch_size=32)
# system.lr = 3e-4

# trainer = Trainer(num_tpu_cores=8,max_epochs=1)   
# trainer = Trainer(max_epochs=1, train_percent_check=0.1)
# checkpoint_callback = ModelCheckpoint(
#     filepath=os.getcwd()+'/checkpoint',
#     verbose=True,
#     monitor='val_loss',
#     mode='min',
#     prefix=''
# )
# trainer = Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=1)
trainer = Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2)
# trainer = Trainer(gpus=1, max_epochs=3, auto_lr_find=True, progress_bar_refresh_rate=1, train_percent_check=0.2)\
# trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2)
# trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2,checkpoint_callback=checkpoint_callback)
# trainer = Trainer(num_tpu_cores=8,max_epochs=1, progress_bar_refresh_rate=1)
# import gc
# gc.collect()

trainer.fit(system)
system.tokenizer.decode(system.train_dataset[0]['source_ids'].squeeze(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
TXT = "query { faculty_aggregate { aggregate { <mask> } } } </s>"
input_ids = system.tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
# logits = system.model(input_ids)[0]
system.tokenizer.decode(system.model.generate(input_ids.cuda())[0])
