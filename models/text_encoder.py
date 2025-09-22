import gc
import torch
import torch.nn.functional as F
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM)
from transformers.models.deberta.modeling_deberta import DebertaOnlyMLMHead
import gc
from tqdm import trange
from utils.functions import get_available_devices
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
LLM_DIM_DICT = {"llama2_7b": 4096,"e5":1024,"BERT":768,"ST":768,"deberta":768}

def mean_pooling(token_embeddings, attention_mask):
    # Efficient pooling by using mask directly
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)


class LLMModel(torch.nn.Module):
    """
    Large Language Model (LLM) for sentence representation, using transformers.
    Supports various pre-trained models including LLaMA, DeBERTa, BERT, etc.
    """

    def __init__(self, llm_name: str, mask_rate: float = 0.5, device: torch.device = torch.device('cuda'),
                 cache_dir: str = "./model_data/model", batch_size: int = 1, max_length: int = 256):
        super().__init__()
        # Check if the model name is valid
        if llm_name not in LLM_DIM_DICT:
            raise ValueError(f"Invalid model name: {llm_name}. Choose from {list(LLM_DIM_DICT.keys())}")

        self.llm_name = llm_name
        self.mask_rate = mask_rate
        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        # Load model and tokenizer
        self.model, self.tokenizer = self.get_llm_model()
        self.model.to(self.device)
        self.config = self.model.config

        # For specific models like BERT and DeBERTa, add a custom classification head
        if self.llm_name in ["deberta", "BERT"]:
            self.cls_head = DebertaOnlyMLMHead(self.model.config).to(self.device)
        else:
            self.cls_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size).to(self.device)
        # Tokenizer padding settings
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'
        self.mask_token_id = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1]

    def get_llm_model(self):
        """
        Loads the appropriate model and tokenizer based on the LLM name.
        """
        model_mapping = {
            "llama2_7b": ("./llm/Llama-2-7b-hf", LlamaForCausalLM, LlamaTokenizer),
            "e5": ("./llm/e5-large-v2", AutoModel, AutoTokenizer),
            "BERT": ("./llm/bert-base-uncased", AutoModel, AutoTokenizer),
            "ST": ("./llm/multi-qa-distilbert-cos-v1", AutoModel, AutoTokenizer),
            "deberta": ("./llm/deberta-v3-base", AutoModel, AutoTokenizer)
        }

        model_name, ModelClass, TokenizerClass = model_mapping.get(self.llm_name, (None, None, None))

        if model_name is None:
            raise ValueError(f"Unknown language model: {self.llm_name}.")

        model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True,
                                                   padding_side="left")

        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token  # Use BOS token as padding for LLaMA

        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

    def forward(self, text_tokens,pooling=False):
        """
        Forward pass to get the [CLS] token embedding from the model.
        """
        text_tokens = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in text_tokens.items()
        }
        emb = self.model(input_ids=text_tokens["input_ids"],
                                attention_mask=text_tokens["attention_mask"],
                                output_hidden_states=True,
                                return_dict=True)["hidden_states"][-1]
        emb = emb.to(torch.float32)
        if pooling:
            emb = self.pooling(emb, text_tokens)
        # Extract [CLS] token embedding
        return emb



    def mask_encoder(self, text_tokens, pooling=False):
        """
        Get the embedding for the masked tokens in the input sequence.
        """
        text_tokens = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in text_tokens.items()
        }
        masked_emb = self.model(input_ids=text_tokens["masked_input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]
        if pooling:
            masked_emb = self.pooling(masked_emb, text_tokens)
        # Extract [CLS] token embedding for masked input
        return masked_emb

    def encode_text(self, texts, to_tensor=True):
        all_text_tokens = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Processing Batches", disable=False):
                sentences_batch = texts[start_index: start_index + self.batch_size]

                text_tokens = self.tokenizer(
                    sentences_batch, return_tensors="pt", padding="longest",
                    truncation=True, max_length=self.max_length
                ).to(self.device)

                text_tokens = {key: val.to(self.device) for key, val in text_tokens.items()}

                masked_input_ids = text_tokens["input_ids"].clone()

                for i in range(masked_input_ids.shape[0]):
                    token_length = (masked_input_ids[i] != self.tokenizer.pad_token_id).sum().item()  # 计算有效 token 数量

                    if token_length > 1:
                        mask_num = max(1,int(token_length * self.mask_rate))  # 计算要 Mask 的数量
                        mask_list = random.sample(range(1, token_length - 1), mask_num)  # 随机选取索引（不 Mask [CLS] 和 [SEP]）
                        masked_input_ids[i, mask_list] = self.mask_token_id  # 替换为 [MASK] 令牌
                text_tokens["masked_input_ids"] = masked_input_ids
                all_text_tokens.append(text_tokens)

        return all_text_tokens




    def flush_model(self):
        """
        Frees up GPU memory by deleting the model and clearing cache.
        """
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()





