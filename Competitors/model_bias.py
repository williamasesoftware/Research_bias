import nltk
nltk.download('punkt')
from gensim.models import Word2Vec

def NLP_model(corpus, name_model):
    corpus_tok = [nltk.word_tokenize(sent) for sent in corpus]
    window_size = 10
    vector_size = 300

    model = Word2Vec(corpus_tok,vector_size=vector_size, window=window_size, min_count=2, workers=4)
    model.save(name_model)

    return model

def word_to_analyze_companies(Word_to_analize,ASW_model,Globant_model,Accenture_model):

    similaresASW = ASW_model.wv.most_similar(Word_to_analize,topn=5)
    similaresGlobant = Globant_model.wv.most_similar(Word_to_analize,topn=5)
    similaresAccenture = Accenture_model.wv.most_similar(Word_to_analize,topn=5)
    print("----------------------------------------------------------------------------------------")
    print("Word: ", Word_to_analize)
    print("ASW: ", similaresASW)
    print("Globant: ",similaresGlobant)
    print("Accenture: ",similaresAccenture)