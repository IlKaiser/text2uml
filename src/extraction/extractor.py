from abc import ABC, abstractmethod

class Extractor(ABC):
    """
        Abstract class for extracting uml schema from text
    """
    @abstractmethod
    def single_extraction(self):
        pass

    @abstractmethod
    def batch_extraction(self):
        pass