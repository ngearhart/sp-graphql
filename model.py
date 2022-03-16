import os

import pytorch_lightning as pl
import torch
from graphqlval import exact_match
# from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
# from transformers import BartTokenizer,BartModel,BartForConditionalGeneration
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)

from dataset import (CoSQLMaskDataset, MaskGraphQLDataset, SpiderDataset,
                     TextToGraphQLDataset)

torch.manual_seed(0)

from constants import DEVICE

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)


class T5MultiSPModel(pl.LightningModule):
    # def __init__(self, train_sampler=None, tokenizer= None, dataset=None, batch_size = 2):
    def __init__(self, hparams, task='denoise', test_flag='graphql', train_sampler=None, batch_size=2, temperature=1.0, top_k=50, top_p=1.0, num_beams=1):
        super(T5MultiSPModel, self).__init__()

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams

        # self.lr=3e-5
        self.save_hyperparameters(hparams)

        self.task = task
        self.test_flag = test_flag
        self.train_sampler = train_sampler
        self.batch_size = batch_size
        # todo load from file if task is finetine.
        if self.task == 'finetune':
            # have to change output_past to True manually
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                't5-base')  # no output past?

        self.model = self.model.to(DEVICE)

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id)
        self.add_special_tokens()

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def add_special_tokens(self):
        # new special tokens
        # the issue could be here, might need to copy.
        special_tokens_dict = self.tokenizer.special_tokens_map
        special_tokens_dict['mask_token'] = '<mask>'
        special_tokens_dict['additional_special_tokens'] = [
            '<t>', '</t>', '<a>', '</a>']
        self.tokenizer.add_tokens(['{', '}', '<c>', '</c>'])
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # For some reason I need this last line. or maybe it had to do with tensorboard

    def _step(self, batch):
        if self.task == 'finetune':
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
            # y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, :].clone()
            lm_labels[y[:, :] == pad_token_id] = -100
            # attention_mask is for ignore padding on source_ids
            # lm_labels need to have pad_token ignored manually by setting to -100
            # todo check the ignore token for forward
            # seems like decoder_input_ids can be removed.
            outputs = self(source_ids, attention_mask=source_mask,
                           lm_labels=lm_labels,)

            loss = outputs[0]

        else:
            y = batch['target_id']
            lm_labels = y[:, :].clone()
            lm_labels[y[:, :] == self.tokenizer.pad_token_id] = -100
            loss = self(
                input_ids=batch["source_ids"],
                lm_labels=lm_labels
            )[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed

        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])

        # creating log dictionary
        tensorboard_logs = {'loss': avg_loss,"Accuracy": correct/total}

        epoch_dictionary={
            # required
            'loss': avg_loss,
            
            # for logging purposes
            'log': tensorboard_logs}

        return epoch_dictionary

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)

        # if self.task == 'finetune':
        #   preds, target = self._generate_step(batch)
        #   accuracy = exact_match.exact_match_accuracy(preds,target)
        #   return {"val_loss": loss, "val_acc": torch.tensor(accuracy) }
        # else:
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # if self.task == 'finetune':
        #   avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        #   tensorboard_logs = {"val_loss": avg_loss, "avg_val_acc": avg_acc}
        #   return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}
        # else:
        tensorboard_logs = {"val_loss": avg_loss}
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, *args, **kwargs):
        if self.trainer.tpu_cores is not None and self.trainer.tpu_cores > 0:
            print("Invalid code (optimizer_step handler)")
            exit(1)
            # xm.optimizer_step(optimizer)
        else:
            optimizer.step(closure=second_order_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def configure_optimizers(self):
        # TODO: Is val_check_interval the same as train_percent_check?
        t_total = len(self.train_dataloader()) * \
            self.trainer.max_epochs * self.trainer.val_check_interval
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in self.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return [optimizer]  # , [scheduler]

    def _generate_step(self, batch):
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            num_beams=self.num_beams,
            max_length=1000,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            # repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [
            self.tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in batch["target_ids"]
        ]
        return (preds, target)

    def test_step(self, batch, batch_idx):
        preds, target = self._generate_step(batch)
        loss = self._step(batch)
        if self.test_flag == 'graphql':
            accuracy = exact_match.exact_match_accuracy(preds, target)
            return {"test_loss": loss, "test_accuracy": torch.tensor(accuracy)}
        else:
            return {"test_loss": loss, "preds": preds, "target": target}

    # def test_end(self, outputs):
    #   return self.validation_end(outputs)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        if self.test_flag == 'graphql':
            avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
            tensorboard_logs = {"test_loss": avg_loss, "test_acc": avg_acc}
            return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}

        else:
            output_test_predictions_file = os.path.join(
                os.getcwd(), "test_predictions.txt")
            with open(output_test_predictions_file, "w+", encoding='utf-8') as p_writer:
                for output_batch in outputs:
                    try:
                        p_writer.writelines(
                            s + "\n" for s in output_batch["preds"])
                    except Exception as e:
                        # Ignore encoding errors
                        print(f'Encoding error: {e}')
            tensorboard_logs = {"test_loss": avg_loss}
            return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}

    def prepare_data(self):
        if self.task == 'finetune':
            self.train_dataset_g = TextToGraphQLDataset(self.tokenizer)
            self.val_dataset_g = TextToGraphQLDataset(
                self.tokenizer, type_path='dev.json')
            self.test_dataset_g = TextToGraphQLDataset(
                self.tokenizer, type_path='dev.json')

            self.train_dataset_s = SpiderDataset(self.tokenizer)
            self.val_dataset_s = SpiderDataset(
                self.tokenizer, type_path='dev.json')
            self.test_dataset_s = SpiderDataset(
                self.tokenizer, type_path='dev.json')

            self.train_dataset = ConcatDataset(
                [self.train_dataset_g, self.train_dataset_s])
            self.val_dataset = ConcatDataset(
                [self.val_dataset_g, self.val_dataset_s])
            # self.test_dataset = ConcatDataset([test_dataset_g, test_dataset_s])
            if self.test_flag == 'graphql':
                self.test_dataset = self.test_dataset_g
            else:
                self.test_dataset = self.test_dataset_s

        else:
            train_dataset_g = MaskGraphQLDataset(self.tokenizer)
            val_dataset_g = MaskGraphQLDataset(
                self.tokenizer, type_path='dev.json')

            train_dataset_s = CoSQLMaskDataset(self.tokenizer)
            val_dataset_s = CoSQLMaskDataset(
                self.tokenizer, type_path='cosql_dev.json')

            self.train_dataset = ConcatDataset(
                [train_dataset_g, train_dataset_s])
            self.val_dataset = ConcatDataset([val_dataset_g, val_dataset_s])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)
