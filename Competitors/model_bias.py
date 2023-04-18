import nltk
nltk.download('punkt')
from gensim.models import Word2Vec

def NLP_model(corpus, name_model):

    """
    Train and save a Word2Vec model using the provided corpus and model name.

    Args:
        corpus (list): List of text documents to train the Word2Vec model.
        name_model (str): The name to use when saving the trained model.

    Returns:
        The trained Word2Vec model.
    """

    corpus_tok = [nltk.word_tokenize(sent) for sent in corpus]
    window_size = 10
    vector_size = 300

    model = Word2Vec(corpus_tok,vector_size=vector_size, window=window_size, min_count=2, workers=4)
    model.save(name_model)

    return model
