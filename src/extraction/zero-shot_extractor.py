"""
    Zero-shot extraction module
""""

from abc import ABC 
from extractor import Extractor
from prompt.zero-shot_prompt import prompt_template
from utils import process_subfolders_with_chain

class ZeroShotExtractor(Extractor, ABC):
    """
        Zero-shot extraction class
    """
    def __init__(self):
        super().__init__()

    def single_extraction(self, text):
        """
            Extract uml schema from text
        """
        pass

    def batch_extraction(self, root_folder):
        """
            Extract uml schema from multiple texts
        """
        pass



from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class ZeroShotExtactorOpenAI(ZeroShotExtractor):
    """
        Zero-shot extraction class using OpenAI
    """
    def __init__(self, model_id="gpt-4o-mini"):
        super().__init()
        self.model = model_id

    def single_extraction(self, text):
        """
            Extract uml schema from text
        """
        model = ChatOpenAI(model=self.model)
        chain = prompt_template | model | StrOutputParser()
        res = chain.invoke({"text":text})
        return res

    def batch_extraction(self, root_folder):
        """
            Extract uml schema from multiple texts
        """
        model_ = ChatOpenAI(model=self.model)
        chain = prompt_template | model_ | StrOutputParser()
        process_subfolders_with_chain(root_folder, chain, type=self.model)

from langchain_ollama import ChatOllama

class ZeroShotExtractorOllama(ZeroShotExtractor):
    """
        Zero-shot extraction class using Ollama
    """
    def __init__(self, model_id):
        super().__init()
        self.model = model_id

    def single_extraction(self, text):
        """
            Extract uml schema from text
        """
        model = ChatOllama(
            model=self.model,
            temperature=0,
        )
        chain = prompt_template | model | StrOutputParser()
        res = chain.invoke({"text": text})
        return res

    def batch_extraction(self, root_folder):
        """
            Extract uml schema from multiple texts
        """
        model_ = ChatOllama(
            model=self.model,
            temperature=0,
        )
        chain = prompt_template | model_ | StrOutputParser()
        process_subfolders_with_chain(root_folder, chain, type=self.model)

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ZeroShotExtractorHuggingface(ZeroShotExtractor):
    """
        Zero-shot extraction class using HuggingFace
    """
    def __init__(self, model_id):
        super().__init__()
        self.model = model_id

    def single_extraction(self, text):
        """
            Extract uml schema from text
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModelForCausalLM.from_pretrained(self.model)
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        chain = prompt_template | ChatHuggingFace(HuggingFacePipeline(pipeline=hf_pipeline)) | StrOutputParser()
        res = chain.invoke({"text": text})
        return res

    def batch_extraction(self, root_folder):
        """
            Extract uml schema from multiple texts
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModelForCausalLM.from_pretrained(self.model)
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        chain = prompt_template | ChatHuggingFace(HuggingFacePipeline(pipeline=hf_pipeline)) | StrOutputParser()
        process_subfolders_with_chain(root_folder, chain, type=self.model)

from langchain_community.llms import MLXPipeline
from langchain_community.chat_models import ChatMLX

class ZeroShotExtractorMLX(ZeroShotExtractor):
    """
        Zero-shot extraction class using MLX
    """
    def __init__(self, model_id):
        super().__init__()
        self.model = model_id

    def single_extraction(self, text):
        """
            Extract uml schema from text
        """
        model = MLXPipeline(model=self.model)
        chain = prompt_template | ChatMLX(model) | StrOutputParser()
        res = chain.invoke({"text": text})
        return res

    def batch_extraction(self, root_folder):
        """
            Extract uml schema from multiple texts
        """
        model = MLXPipeline(model=self.model)
        chain = prompt_template | ChatMLX(model) | StrOutputParser()
        process_subfolders_with_chain(root_folder, chain, type=self.model)