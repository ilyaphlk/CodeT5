from ttadapters.adapters import ADAPTER_CONFIG_MAPPING
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from transformers import Seq2SeqTrainingArguments

@dataclass
class AdapterTrainingArguments:
    """Defines the adapters parameters."""
    train_task_adapters: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds task adapters in the model."})

    adapter_config_name: Optional[str] = field(
        default="adapter", metadata={"help": "config name for the adapter layers, should be selected "
        f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."}
    )

    add_layer_norm_before_adapter: Optional[bool] = field(default=False, metadata={
        "help": "whether to have layer-norm before adapter."})

    add_layer_norm_after_adapter: Optional[bool] = field(default=True,
        metadata={"help": "whether to have layer-norm after adapter."})

    hidden_dim: Optional[int] = field(default=128, metadata={"help": "defines the default hidden dimension for "
        "adapter layers."})

    task_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "adapter layers."})

    non_linearity: Optional[str] = field(default="swish", metadata={"help": "Defines nonlinearity for adapter layers."})

    unfreeze_lm_head: bool = field(default=False, metadata={"help": "If set unfreeze the last linear layer."})

    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "If set, unfreezes the layer norms."})

    task_adapter_layers_encoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the encoder."})

    task_adapter_layers_decoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the decoder."})

    task_adapter_in_decoder: Optional[bool] = field(default=True, metadata={"help": "If set to false, do not include"
                                                                                    "task adapters in the decoder."})

    hypercomplex_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the hypercomplex layers"
                                                                                "for adapters."})

    hypercomplex_division: Optional[int] = field(default=8, metadata={"help": "Defines the number to divide the dimensions"
                                                                              "of the linear layer by it."})

    intrinsic_model: Optional[bool] = field(default=False, metadata={"help": "If set, computes all parameters of the "
                                                                             "model with an intrinsic vector."})

    intrinsic_said: Optional[bool] = field(default=False, metadata={"help": "If set, computes the SAID version of the"
                                                                            "model with intrinsic vector."})

    intrinsic_dim: Optional[int] = field(default=100, metadata={"help": "Defines the intrinsic dimensionality."})

    normalize_intrinsic_projections: Optional[bool] = field(default=False, metadata={"help": "If set, normalizes "
        "the intrinsic projection matrices."})

    intrinsic_projection: Optional[str] = field(default="fastfood", metadata={"help": "Defines the type of projection"
        "for intrinsic adapters, it can be random or fastfood."})

    learn_phm: Optional[bool] = field(default=True, metadata={"help": "If set, learns the phm rules in Hypercomplex adapters."})
    normalize_phm_weight: Optional[bool] = field(default=False, metadata={"help": "Weather to normalize the weights of"
                                                                                  "the PHM layer."})

    intrinsic_layer_norms: Optional[bool] = field(default=False, metadata={"help": "If selected, then in case of unfreezing"
        " layernorms for intrinsic_adapters case, it also adds the layernorms parameters inside the parameters given for"
        " the intrinsic projection, and if this is not set, those parameters are not projected with intrinsic vector."})

    hypercomplex_nonlinearity: Optional[str] = field(default="glorot-uniform", metadata={"help": "Defines the nonlinearity for the"
        " hypercomplex adapter layers."})

    shared_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, uses a shared phm rules for all"
        " hypercomplex adapter layers."})

    factorized_phm: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the weights for the W in"
        " hypercomplex adapters."})

    shared_W_phm: Optional[bool] = field(default=False, metadata={"help": "If set, shares the W in phm adapter layers between all adapters."})

    factorized_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the shared weights for the W in"
        " hypercomplex adapters."})

    phm_c_init: Optional[str] = field(default="normal", metadata={"help": "Initialization for the phm rules."})

    phm_rank: Optional[int] = field(default=1, metadata={"help":"sets the rank for the phm decomposition."})

    phm_init_range: Optional[float] = field(default=0.01, metadata={"help": "defines the phm init range."})

    add_adapter_in_feed_forward: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the feedforward."})

    add_adapter_in_self_attention: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the selfattention"})

    prefix_tuning: Optional[bool] = field(default=False, metadata={"help": "If set, uses prefix tuning."})

    prefix_dim: Optional[int] = field(default=100, metadata={"help": "Specifies the prefix embedding dimension."})

    init_prefix_from_vocab: Optional[bool] = field(default=False, metadata={"help": "Initialize prefix from the tokens of pretrained t5-base model."})

    kronecker_prod: Optional[bool] = field(default=False, metadata={"help": "If set, compute the kronecker using another version."})

    bitfit: Optional[bool] = field(default=False, metadata={"help": "If set, we train the bitfit model."})

    freeze_bitfit_lm_head: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})

    freeze_bitfit_lm_head_all: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})

    # Low-rank adapters.
    low_rank_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the low-rank adapters."})

    low_rank_w_init: Optional[str] = field(
        default="glorot-uniform",
        metadata={"help": "Defines the initialization for low-rank adapters."}
    )

    low_rank_rank: Optional[int] = field(
        default=1,
        metadata={"help": "Defines the rank of low-rank adapters."}
    )

    # Tensor-Train adapters

    tensor_train_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the Tensor-Train adapters."})
    tensor_train_single: Optional[bool] = field(default=False, metadata={"help": "If set, uses non-bottleneck scheme for Tensor-Train adapters."})

    tt_rank: Optional[int] = field(
        default=8,
        metadata={"help": "Defines the rank of tensor-train adapters."}
    )

    tt_d: Optional[int] = field(
        default=3,
        metadata={"help": "Defines the number of cores in tensor-train adapters."}
    )

    tt_shape: Optional[List[List[int]]] = field(
        default=None,
        metadata={"help": "If set, explicitly defines the shape of a tensor-train linear layer"}
    )

    reverse_out_shape: bool = field(
        default=False,
        metadata={"help": "Whether or not to reverse out_dim quantization for Tensor Train Linear layer"}
    )

    factorize_smaller_dim: bool = field(
        default=True,
        metadata={"help": "Whether or not to factorize smaller dim for Tensor Train Linear layer"}
    )

    cores_nonlinearity: str = field(
        default=None,
        metadata={"help": "Type of nonlinearity to add between Tensor Train cores"}
    )

    use_scripted_mul: bool = field(
        default=False,
        metadata={"help": "If set, uses torch.jit to multiply Tensor Train cores"}
    )

    naive: bool = field(
        default=False,
        metadata={"help": "If set, uses naive premultiplication of ttcores before multiplying with input. Overrides use_scripted_mul"}
    )

    use_checkpointing: bool = field(
        default=False,
        metadata={"help": "If set, uses gradient checkpointing in adapter layers"}
    )

    ttcore_checkpointing: bool = field(
        default=False,
        metadata={"help": "If set, uses gradient checkpointing after each ttcore inside TTLinear layer"}
    )

    auto_shape_mode: str = field(
        default='ascending',
        metadata={"help": "shape factorization mode"}
    )

    use_ScaleNorm: bool = field(
        default=False,
        metadata={"help": "if set, uses Scalar Layer Normalization (as proposed in Transformers Without Tears paper)"}
    )

    ScaleNorm_scale: float = field(
        default=1.0,
        metadata={"help": "scale factor for ScaleNorm"}
    )

    use_TTLayerNorm: bool = field(
        default=False,
        metadata={"help": "use TTLinear layer to represent LayerNorm vector"}
    )

    TTLayerNorm_rk: int = field(
        default=2,
        metadata={"help": "rk to use when tensorising LN vectors"}
    )

    TTLayerNorm_preinit: bool = field(
        default=False,
        metadata={"help": "if set, approximates pretrained LayerNorm weights instead of const init"}
    )

    use_LayerNorm_mean: bool = field(
        default=False,
        metadata={"help": "if set, approximates with mean of pretrained LayerNorm weights"}
    )

    use_bias: bool = field(
        default=True,
        metadata={"help": "if set, uses bias in adapter layers"}
    )

    use_TTBias: bool = field(
        default=False,
        metadata={"help": "if set, tensorises bias in adapter layers"}
    )

    TTBias_rk: int = field(
        default=2,
        metadata={"help": "rk to use when tensorising bias vectors"}
    )

    use_LoRA: bool = field(
        default=False,
        metadata={"help": "if set, use LoRA-style adapters"}
    )

    lora_dense: bool = field(
        default=False,
        metadata={"help": "use regular linear layer as lora part for debugging purposes"}
    )

    use_TTLoRA: bool = field(
        default=False,
        metadata={"help": "if set, use LoRA-style TT-adapters"}
    )

    ttlora_separate: bool = field(
        default=False,
        metadata={"help": "use TTlinear for LoraA and LoraB parts separately"}
    )

    TTLoRA_init: str = field(
        default=None,
        metadata={"help": "if None, uses glorot initialization for TTLoRA. Kaiming imitates the LoRA initialization, described in the original paper, by initializing the first TTCore with Kaiming initialization"}
    )


    # additional fields

    freeze_cores: Optional[str] = field(
        default=None,
        metadata={"help": "If set, defines the strategy to freeze TT-cores for fine-tuning across tasks"}
    )

    task_expansion_factor: Optional[int] = field(
        default=0,
        metadata={"help": "defines the expansion factor for "
        "adapter layers. if set to zero, adapter operates in reduction mode."}
    )

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                                 "the model."})
    do_test: Optional[bool] = field(default=False, metadata={"help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(default=False, metadata={"help": "if set, measures the memory"})
    prefix_length: Optional[int] = field(default=100, metadata={"help": "Defines the length for prefix tuning."})
    experiment_name: Optional[str] = field(default="default_name", metadata={"help": "Name used for experimentation."})
    project_name: Optional[str] = field(default="TTAdapter", metadata={"help": "Project name for wandb."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the evaluation dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    test_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the test dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."}
    )
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    data_seed: Optional[int] = field(default=42, metadata={"help": "seed used to shuffle the data."})

    def __post_init__(self):
        if self.task_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.test_max_target_length is None:
            self.test_max_target_length = self.max_target_length