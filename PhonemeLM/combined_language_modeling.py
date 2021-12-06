import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    CombinedMonolingualDataset, # TODO
    LMContextWindowDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II


@dataclass
class CombinedLanguageModelingConfig(FairseqDataclass):
    n_sequence_type: int = field(
        default=2, metadata={"help": "number of sequence types"}
    )
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task('combined_language_modeling', dataclass=CombinedLanguageModelingConfig)
class CombinedLanguageModelingTask(LegacyFairseqTask):
    """
    Train a language model

    Args:
        dictionary ()
    """

    def __init__(self, args, dictionaries, output_dictionaries=None, targets=None):
        super().__init__(args)
        self.n_seq_type = args.n_sequence_type
        print('Number of sequence types: ', self.n_seq_type) # XXX
        self.dictionaries = dictionaries
        self.output_dictionaries = output_dictionaries or dictionaries

        if targets is None:
            targets = ["future"]
        self.targets = targets
    
    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionaries = []
        output_dictionaries = []
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            for i in range(args.n_sequence_type):
                dictionary = Dictionary.load(os.path.join(paths[0], str(i), "dict.txt")) # TODO
                logger.info("dictionary: {} types".format(len(dictionary)))
                output_dictionary = dictionary
                if args.output_dictionary_size >= 0:
                    output_dictionary = TruncatedDictionary(
                        dictionary, args.output_dictionary_size
                    )
                dictionaries.append(dictionary)
                output_dictionaries.append(output_dictionaries)
        return (dictionaries, output_dictionaries)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionaries, output_dictionaries = cls.setup_dictionary(args, **kwargs)
        
        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionaries, output_dictionaries, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def build_criterion(self, args): # TODO

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> CombinedMonolingualDataset:
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        datasets = []
        for i in range(self.n_seq_type):
          split_path = os.path.join(data_path, split, str(i))

          # each process has its own copy of the raw data (likely to be an np.memmap)
          dataset = data_utils.load_indexed_dataset(
              split_path, self.dictionaries[i], self.args.dataset_impl, combine=combine
          )
          if dataset is None:
              raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")

          datasets.append(dataset)
        dataset = self.combine_dataset(datasets)
      
        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = (
                self.args.batch_size_valid if "valid" in split else self.args.batch_size
            )

        self.datasets[split] = CombinedMonolingualDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

    def combine_dataset(self, datasets):
        dataset = []
        for i in range(len(datasets[0])):
            dataset.append([[] for _ in range(len(datasets[0][0]))])
            for j in range(len(datasets)):
                for k in range(len(datasets[j][i]))
                    dataset[-1][k].append(datasets[j][i][k])
        return dataset

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        print('Warning: source_dictionary() is called') # XXX Need choice to select which sequence type 
        return self.dictionaries[0]

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        print('Warning: target_dictionary() is called') # XXX Need choice to select which sequence type 
        return self.output_dictionaries[0]
