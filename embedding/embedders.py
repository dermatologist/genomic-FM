# https://github.com/frederikkemarin/BEND/blob/e1e63faec4dddf7c8c63fd9f419b87f535be6355/bend/utils/embedders.py
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

from models.dnabert2 import BertModel as DNABert2BertModel
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
                   model_path: str = '/home/orion-lab/data/genome/codon-bert/',
                   model_max_length: int = 512,
                     **kwargs):
        if not os.path.exists(model_path):
            print(f'Path {model_path} does not exists, check if the wrong path was given. If not download from https://github.com/jerryji1993/DNABERT')

        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = GenomicTokenizer(model_max_length)
        self.model = AutoModel.from_pretrained(model_path, config=config)
        self.model.to(device)
        self.model.eval()

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        embeddings = []
        with torch.no_grad():
            for seq in tqdm(sequences, disable=disable_tqdm):
                input_ids = torch.tensor(self.tokenizer(seq)["input_ids"]).unsqueeze(0)
                attention_mask = torch.tensor(self.tokenizer(seq)["attention_mask"]).unsqueeze(0)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                embeddings.append(embedding.detach().cpu().numpy())
        return embeddings

class DNABert2Embedder(BaseEmbedder):
    """
    Embed using the DNABERT2 model https://arxiv.org/pdf/2306.15006.pdf
    """
    def load_model(self, model_name = "zhihan1996/DNABERT-2-117M", **kwargs):
        """
        Load the DNABERT2 model.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "zhihan1996/DNABERT-2-117M".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.
        """


        # keep the source in this repo to avoid using flash attn.
        self.model = DNABert2BertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)

        # https://github.com/Zhihan1996/DNABERT_2/issues/2
        self.max_length = 10000 #nucleotides.


    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        '''Embeds a list sequences using the DNABERT2 model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        '''
        # '''
        # Note that this model uses byte pair encoding.
        # upsample_embedding repeats BPE token embeddings so that each nucleotide has its own embedding.
        # The [CLS] and [SEP] tokens are removed from the output if remove_special_tokens is True.
        # '''
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):

                chunks = [sequence[chunk : chunk + self.max_length] for chunk in  range(0, len(sequence), self.max_length)] # split into chunks

                embedded_chunks = []
                for n_chunk, chunk in enumerate(chunks):
                    #print(n_chunk)

                    input_ids = self.tokenizer(chunk, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                    #print(input_ids.shape)
                    output = self.model(input_ids.to(device))[0].detach().cpu().numpy()

                    if upsample_embeddings:
                        output = self._repeat_embedding_vectors(self.tokenizer.convert_ids_to_tokens(input_ids[0]), output)

                    # for intermediate chunks the special tokens need to go.
                    # if we only have 1 chunk, keep them for now.
                    if len(chunks) != 1:
                        if n_chunk == 0:
                            output = output[:,:-1] # no SEP
                        elif n_chunk == len(chunks) - 1:
                            output = output[:,1:] # no CLS
                        else:
                            output = output[:,1:-1] # no CLS and no SEP

                    embedded_chunks.append(output)

                embedding = np.concatenate(embedded_chunks, axis=1)

                if remove_special_tokens:
                    embedding = embedding[:,1:-1]

                embeddings.append(embedding)


        return embeddings



    # GATTTATTAGGGGAGATTTTATATATCCCGA
    # ['[CLS]', 'G', 'ATTTATT', 'AGGGG', 'AGATT', 'TTATAT', 'ATCCCG', 'A', '[SEP]']
    @staticmethod
    def _repeat_embedding_vectors(tokens: Iterable[str], embeddings: np.ndarray, has_special_tokens: bool = True):
        '''
        Byte-pair encoding merges a variable number of letters into one token.
        We need to repeat each token's embedding vector for each letter in the token.
        '''
        assert len(tokens) == embeddings.shape[1], 'Number of tokens and embeddings must match.'
        new_embeddings = []
        for idx, token in enumerate(tokens):

            if has_special_tokens and (idx == 0 or idx == len(tokens) - 1):
                new_embeddings.append(embeddings[:, [idx]]) # (1, 768)
                continue
            token_embedding = embeddings[:, [idx]] # (1, 768)
            if token == '[UNK]':
                new_embeddings.extend([token_embedding])
            else:
                new_embeddings.extend([token_embedding] * len(token))

        # list of (1,1, 768) arrays
        new_embeddings = np.concatenate(new_embeddings, axis=1)
        return new_embeddings