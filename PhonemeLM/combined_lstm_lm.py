import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqLanguageModel,
    BaseFairseqModel,
)
from .lstm_lm import LSTMLanguageModel
from .lstm import LSTMDecoder


DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model("combined_lstm_lm")
class CombinedLSTMLanguageModel(FairseqLanguageModel):
   
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--residuals', default=False,
                            action='store_true',
                            help='applying residuals between LSTM layers')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, "max_target_positions", None) is not None:
            max_target_positions = args.max_target_positions
        else:
            max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        decoders = [LSTMDecoder(
            dictionary=task.dictionaries[i],
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=False,  # decoder-only language model doesn't support attention
            encoder_output_units=0,
            pretrained_embed=None,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            residuals=args.residuals,
        ) for i in range(args.n_sequence_type)]
            
        return cls(decoders)

    def __init__(self, decoders):
        """
        Args:
            decoders: 
        """
        super().__init__(decoders[0])
        self.decoders = nn.ModuleList(decoders)
        out_embed_dim = sum(decoder.embed_tokens.weight.size(1) for decoder in decoders) 
        nums_embeddings = [decoder.embed_tokens.weight.size(0) for decoder in decoders]
        self.fc_outs = nn.ModuleList([nn.Linear(out_embed_dim, num_embeddings) for num_embeddings in nums_embeddings])        
        
    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens: LongTensor of shape `(batch, tgt_len, n_sequence_type)` 
            src_lengths: LongTensor of shape `(batch)`

        Returns:
            out:
                - the decoder's output of shape `(batch, seq_len, vocab, n_sequence_type)`
        """
        return self.output_layer(self.extract_features(src_tokens))

    def extract_features(self, src_tokens):
        src_tokens_list = torch.split(src_tokens, 1, dim=-1)
        features = [decoder.extract_features(src.squeeze(-1))[0] for decoder, src in zip(self.decoders, src_tokens_list)]
        return torch.cat(features, dim=-1)
    
    def output_layer(self, features):
        outs = [fc_out(features) for fc_out in self.fc_outs]
        return outs

    def get_normalized_probs(self, net_output, log_probs=True):
        logits = net_output.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
        

@register_model_architecture("combined_lstm_lm", "combined_lstm_lm")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "0")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )
    args.residuals = getattr(args, "residuals", False)
