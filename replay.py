#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.
"""
from torch.utils.data import DataLoader
from torch.nn import DataParallel as DP
from torch.utils.data import IterableDataset
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from hyperpyyaml import resolve_references
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DistributedSamplerWrapper
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from collections import OrderedDict

import os
import sys
import torch
import torchaudio
import random
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from create_replay_dataset import create_double_dataset
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import numpy as np

torchaudio.set_audio_backend("soundfile")

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig1
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Downsample the inputs if specified
        # Handling SpeechBrain vs HuggingFance pretrained models
        feats = self.modules.wav2vec2(wavs)

        x = self.modules.enc(feats)

        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        if stage == sb.Stage.VALID or (
            stage == sb.Stage.TEST and not self.hparams.use_language_modelling
        ):

            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        if self.hparams.epoch_counter.current  <= self.hparams.warming_epochs : 
            ssl = False
            teacher_feats = None
            return ssl, p_ctc, wav_lens, p_tokens, feats, teacher_feats
        else : 
            if random.random()< self.hparams.ssl_probability and stage==sb.Stage.TRAIN : 
                wavs, wav_lens = batch.sig2 # Loading the replay dataset
                #Running the replay loss
                wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)



                if self.hparams.augment : 
                    teacher_wavs = wavs
                    for count, augment in enumerate(self.hparams.augment_pipeline):
                        teacher_wavs = augment(teacher_wavs, wav_lens)
                else : 
                    teacher_wavs = wavs 
                teacher_feats = self.modules.teacher(teacher_wavs).detach()
                teacher_feats = torch.mean(teacher_feats[-8:], axis = 0)
                extract_features = self.modules.wav2vec2.model.feature_extractor(wavs)
                extract_features = extract_features.transpose(1, 2)

                hidden_states, extract_features = self.modules.wav2vec2.model.feature_projection(extract_features)
                hidden_states = self.modules.wav2vec2.model.train_mask_hidden_states(
                    hidden_states, mask_time_indices=None, attention_mask=None
                )
                encoder_outputs = self.modules.wav2vec2.model.encoder(
                    hidden_states,
                    attention_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                )

                hidden_states = encoder_outputs[0]
                feats = hidden_states
                ssl = True
            else : 
                ssl = False 
                teacher_feats = None
            return ssl, p_ctc, wav_lens, p_tokens, feats, teacher_feats



    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        ssl, p_ctc, wav_lens, predicted_tokens, feats, teacher_feats = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        self.downstream_loss.append(loss.detach().cpu().item())
        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
    
        if ssl : 
            ssl_loss = self.hparams.ssl_loss(feats, teacher_feats)
            loss += ssl_loss
            self.ssl_loss.append(ssl_loss.detach().cpu().item())
        
        return ssl, loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                if not self.hparams.freeze_wav2vec:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            with self.no_sync():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            ssl,loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        return loss.detach().cpu()
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            ssl, loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)
            if self.hparams.change_train and self.hparams.epoch_counter.current % self.hparams.change_train_epochs == 0 :
                #Create the new csv
                name = os.path.join(self.hparams.save_folder, f"epoch_{self.hparams.epoch_counter.current}_train.csv")
                create_double_dataset(self.hparams.replay_dataset,self.hparams.train_dataset,name)
                hparams["train_csv"] =name 
                train_data= dataio_prepare_change_train(
                    hparams, self.tokenizer
                )
                logger.info("changing the dataset")
                train_set = self.make_dataloader(
                    train_data, stage=sb.Stage.TRAIN, **train_loader_kwargs
                )

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break



    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
        self.downstream_loss = []
        self.ssl_loss = []


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            if len(self.ssl_loss)>0 : 

                self.train_stats["ssl_loss"] = np.mean(self.ssl_loss)
            self.train_stats["downstream_loss"] = np.mean(self.downstream_loss)
 
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_checkpoint()
            if self.hparams.changing_teacher and self.hparams.epoch_counter.current > self.hparams.warming_epochs : 
                # Update the teacher
                student_params = OrderedDict(self.modules.wav2vec2.named_parameters())
                teacher_params = OrderedDict(self.modules.teacher.named_parameters())

                # check if both model contains the same set of keys
                assert student_params.keys() == teacher_params.keys()

                for name, param in teacher_params.items():
                    teacher_params[name] -= self.hparams.decay * (teacher_params[name] - student_params[name]) 

                self.modules.teacher.load_state_dict(teacher_params)
                logger.info("Updating the teacher")


        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.encoder_wrapper.parameters()
            )

        else:  # HuggingFace pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)

def dataio_prepare_change_train(hparams, label_encoder):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )


    datasets = [train_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "wav2")
    @sb.utils.data_pipeline.provides("sig1", "sig2")
    def audio_pipeline(wav, wav2):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        sig1 = resampled
        info = torchaudio.info(wav2)
        sig2 = sb.dataio.dataio.read_audio(wav2)
        sig2 = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig2)

        return sig1, sig2

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list",  "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig1","sig2", "wrd", "char_list",  "tokens"],
    )
    return train_data 
def dataio_prepare_train(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )


    datasets = [train_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "wav2")
    @sb.utils.data_pipeline.provides("sig1", "sig2")
    def audio_pipeline(wav, wav2):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        sig1 = resampled
        info = torchaudio.info(wav2)
        sig2 = sb.dataio.dataio.read_audio(wav2)
        sig2 = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig2)

        return sig1, sig2

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list",  "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
        "unk_label": hparams["unk_index"]
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig1","sig2", "wrd", "char_list",  "tokens"],
    )
    return train_data ,label_encoder


def dataio_prepare_eval(hparams, label_encoder):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]


    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [ valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig1")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list",  "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig1", "wrd", "char_list",  "tokens"],
    )
    return  valid_data, test_datasets




if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    """
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )
    """
    # here we create the datasets objects as well as tokenization and encoding
    name = os.path.join(hparams["save_folder"], f"epoch_1_train.csv")
    create_double_dataset(hparams["replay_dataset"],hparams["train_dataset"],name)
    hparams["train_csv"] =name 


    # here we create the datasets objects as well as tokenization and encoding
    train_data, label_encoder = dataio_prepare_train(
        hparams
    )

    # Loading the labels for the LM decoding and the CTC decoder
    if hasattr(hparams, "use_language_modelling"):
        if hparams["use_language_modelling"]:
            try:
                from pyctcdecode import build_ctcdecoder
            except ImportError:
                err_msg = "Optional dependencies must be installed to use pyctcdecode.\n"
                err_msg += "Install using `pip install kenlm pyctcdecode`.\n"
            raise ImportError(err_msg)

            ind2lab = label_encoder.ind2lab
            labels = [ind2lab[x] for x in range(len(ind2lab))]
            labels = [""] + labels[
                1:
            ]  # Replace the <blank> token with a blank character, needed for PyCTCdecode
            decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=hparams["ngram_lm_path"],  # .arpa or .bin
                alpha=0.5,  # Default by KenLM
                beta=1.0,  # Default by KenLM
            )
    else:
        hparams["use_language_modelling"] = False

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = label_encoder
    asr_brain.tokenizer.add_unk()

    print(asr_brain.tokenizer.lab2ind)
    valid_data, test_datasets = dataio_prepare_eval(
        hparams, label_encoder
    )


    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
