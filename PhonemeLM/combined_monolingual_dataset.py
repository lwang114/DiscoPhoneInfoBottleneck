import numpy as np
import torch

from fairseq import FairseqDataset, data_utils


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                  res.append(
                      data_utils.collate_tokens(
                          [s[key][i] for s in samples],
                          pad_idx,
                          eos_id,
                          left_pad=False
                      )
                  )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
                pad_to_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    src_tokens_list = merge("source", is_list=True)
    src_tokens = torch.stack(src_tokens_list, dim=-1)
    target = src_tokens

    return {
        "id": torch.LongTensor([s["id"] for s in samples]), 
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"][0]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens, 
            "src_lengths": torch.LongTensor([s["source"][0].numel() for s in samples]), 
        },
        "target": target,
    }


class CombinedMonolingualDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        data_path: str, path to the data files after running fairseq_preprocess
        vocabs (List[~fairseq.data.Dictionary]): list of vocabulary for each sequence type
    """

    def __init__(
        self,
        dataset,
        sizes,
        src_vocabs, 
        tgt_vocabs=None,
        add_eos_for_other_targets=False,
        shuffle=False,
        targets=None,
        add_bos_token=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
        src_lang_idx=None,
        tgt_lang_idx=None
    ): 
        self.targets = 'future'
        self.dataset = dataset 
        self.sizes = np.asarray(sizes)
        self.vocab = src_vocabs
        self.tgt_vocabs = tgt_vocabs or src_vocabs
        self.num_seq_type = len(src_vocabs)
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz
        self.src_lang_idx = src_lang_idx
        self.tgt_lang_idx = tgt_lang_idx

        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets

    def __getitem__(self, index):
        source, future_target, past_target = self.dataset[index]
        source, target = self._make_source_target(
            source, future_target, past_target
        )
        source, target = self._maybe_add_bos(source, target) 
        return {"id": index, "source": source, "target": target}

    def __len__(self):
        return len(self.dataset)

    def _maybe_add_bos(self, source, target):
        if self.add_bos_token:
            print('add bos token') # XXX
            source = torch.stack([torch.cat([source[i].new([self.vocabs[i].bos()]), source[i]])
                                  for i in range(self.num_seq_type)], dim=-1)
            if target is not None:
                target = torch.stack([torch.cat([target.new([self.tgt_vocabs[i].bos()]), target[i]])
                                      for i in range(self.num_seq_type)], dim=-1) 
        return source, target

    def _make_source_target(self, source, future_target, past_target):
        if self.targets is not None:
            target = []

            if (
                self.add_eos_for_other_targets
                and (("self" in self.targets) or ("past" in self.targets))
                and source[-1] != self.vocab.eos()
            ):
                # append eos at the end of source
                source = torch.stack([torch.cat([source[i], source.new([self.vocabs[i].eos()]), source[i]])
                                      for i in range(self.num_seq_type)], dim=-1)

                if "future" in self.targets:
                    future_target = torch.stack([torch.cat(
                        [future_target[i], future_target[i].new([self.vocabs[i].pad()])]
                    ) for i in range(self.num_seq_type)], dim=-1)
                if "past" in self.targets:
                    # first token is before the start of sentence which is only used in "none" break mode when
                    # add_eos_for_other_targets is False
                    past_target = torch.stack([torch.cat(
                        [
                            past_target[i].new([self.vocabs[i].pad()]),
                            past_target[i][1:],
                            source[i][-2, None],
                        ]
                    ) for i in range(self.num_seq_type)], dim=-1)

            for t in self.targets:
                if t == "self":
                    target.append(source)
                elif t == "future":
                    target.append(future_target)
                elif t == "past":
                    target.append(past_target)
                else:
                    raise Exception("invalid target " + t)

            if len(target) == 1:
                target = target[0]
        else:
            target = future_target

        return source, target

    def collator(self, samples):
        """
        Merge a list of samples to form a mini-batch, as in 
        https://github.com/pytorch/fairseq/blob/main/fairseq/data/monolingual_dataset.py

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:
                
                - `id` (LongTensor): example IDs in the original
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  
                  - `src_tokens` (LongTensor): a padded 3D Tensor of tokens in 
                    the source sentence of shape `(bsz, src_len, n_seq_type)`. Padding will
                    appear on the right
                
                - `target` (LongTensor): a padded 3D Tensor of tokens in the 
                  target sentence of shape `(bsz, tgt_len, n_seq_type)`. Padding will appear
                  on the right. 
        """
        return collate(
            samples,
            self.vocab.pad(),
            self.vocab.eos(),
        )
