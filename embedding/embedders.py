'''
embedders.py
------------
Wrapper classes for embedding sequences with pretrained DNA language models using a common interface.
The wrapper classes handle loading the models and tokenizers, and embedding the sequences. As far as possible,
models are downloaded automatically.
They also handle removal of special tokens, and optionally upsample the embeddings to the original sequence length.

Embedders can be used as follows. Please check the individual classes for more details on the arguments.

``embedder = EmbedderClass(model_name, some_additional_config_argument=6)``

``embedding = embedder(sequence, remove_special_tokens=True, upsample_embeddings=True)``

'''



import torch
import numpy as np
from typing import List, Iterable
from functools import partial
import os

# from bend.models.awd_lstm import AWDLSTMModelForInference
# from bend.models.dilated_cnn import ConvNetModel
# from bend.models.gena_lm import BertModel as GenaLMBertModel
# from bend.models.hyena_dna import HyenaDNAPreTrainedModel, CharacterTokenizer
# from bend.models.dnabert2 import BertModel as DNABert2BertModel
# from bend.utils.download import download_model, download_model_zenodo
from genomic_tokenizer import GenomicTokenizer

from tqdm.auto import tqdm
from transformers import logging, BertModel, BertConfig, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer, BigBirdModel, AutoModelForMaskedLM
from sklearn.preprocessing import LabelEncoder
logging.set_verbosity_error()



# TODO graceful auto downloading solution for everything that is hosted in a nice way
# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


##
## GPN https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1
##

class BaseEmbedder():
    """Base class for embedders.
    All embedders should inherit from this class.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the embedder. Calls `load_model` with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments. Passed to `load_model`.
        **kwargs
            Keyword arguments. Passed to `load_model`.
        """
        self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Load the model. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def embed(self, sequences:str, *args, **kwargs):
        """Embed a sequence. Should be implemented by the inheriting class.

        Parameters
        ----------
        sequences : str
            The sequences to embed.
        """
        raise NotImplementedError

    def __call__(self, sequence: str, *args, **kwargs):
        """Embed a single sequence. Calls `embed` with the given arguments.

        Parameters
        ----------
        sequence : str
            The sequence to embed.
        *args
            Positional arguments. Passed to `embed`.
        **kwargs
            Keyword arguments. Passed to `embed`.

        Returns
        -------
        np.ndarray
            The embedding of the sequence.
        """
        return self.embed([sequence], *args, disable_tqdm=True, **kwargs)[0]


class CodonBertEmbedder(BaseEmbedder):
    '''Embed using the CodonBert model '''

    def load_model(self,
                   model_path: str = '~/data/genome/codonbert/',
                   model_max_length: int = 512,
                     **kwargs):
        if not os.path.exists(model_path):
            print(f'Path {model_path} does not exists, check if the wrong path was given. If not download from https://github.com/jerryji1993/DNABERT')

        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = GenomicTokenizer(model_max_length)
        self.bert_model = AutoModel.from_pretrained(model_path, config=config)
        self.bert_model.to(device)
        self.bert_model.eval()

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        embeddings = []
        with torch.no_grad():
            for seq in tqdm(sequences, disable=disable_tqdm):
                input_ids = self.tokenizer(seq)["input_ids"]
                input_ids = input_ids.to(device)
                embedding = self.model(input_ids=input_ids).last_hidden_state
                embeddings.append(embedding.detach().cpu().numpy())
        return embeddings

