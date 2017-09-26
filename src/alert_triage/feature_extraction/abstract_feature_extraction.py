
from abc import ABCMeta, abstractmethod

class AbstractFeatureExtraction:
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_model(self): pass
    """ Builds a model if that is part of the feature extractor method 
    
        This method must be implemented by all feature extraction classes.
        If there is not associated model, then use "pass"

    """

    @abstractmethod
    def extract_features(self, alert): pass
    """ Extracts features for the given alert 
   
        This method must be implemented by all feature extraction classes.
    
        Args:
        alert The alert to be processed.  The datatype is a dictionary.
    
    """

    @abstractmethod
    def pickle(self, handle): pass
    """ Custom pickle function.

        Some of the extractors can't be pickled directly, so this 
        provides a way to pickle them in a custom way.

        Arguments:
        handle - The handle to a pickle file that has already been opened
                 with "wb"
    """

    @abstractmethod
    def unpickle(self, handle): pass
    """ Custom pickle function.

        Some of the extractors can't be pickled directly, so this 
        provides a way to pickle them in a custom way.

        Arguments:
        handle - The handle to a pickle file that has already been opened
                 with "wb"
    """

