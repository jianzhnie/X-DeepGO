from .deepgoplus import DeepGOPlusModel
from .esm_model import MLP, EsmTransformer
from .modeling_lstm import (ContrastiveProteinLSTMModel,
                            MultilabelProteinLSTMModel, ProteinLSTMConfig)
from .label_wise_attention import LabelWiseAttentionModel


def get_model(args):
    if args.model == 'deepgoplus':
        model = DeepGOPlusModel(args.num_labels)

    if args.model == 'esm':

        model = EsmTransformer(num_labels=args.num_labels,
                               pool_mode=args.pool_mode,
                               fintune=False)
    if args.model == 'esm_embedding':

        model = MLP(input_size=1280, num_labels=args.num_labels)

    if args.model == 'bert_embedding':

        model = MLP(input_size=768, num_labels=args.num_labels)

    if args.model == 'lstm':

        config = ProteinLSTMConfig(num_labels=args.num_labels)
        model = MultilabelProteinLSTMModel(config)

    if args.model == 'contrastive_lstm':

        config = ProteinLSTMConfig()
        model = ContrastiveProteinLSTMModel(config)
    
    if args.model == 'label_wise_attention':
        model = LabelWiseAttentionModel()

    return model
