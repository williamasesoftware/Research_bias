import nltk
nltk.download('punkt')
from gensim.models import Word2Vec

def NLP_model(corpus, name_model):
    corpus_tok = [nltk.word_tokenize(sent) for sent in corpus]

    # Define los parámetros de tu modelo personalizado
    vector_size = 10
    window_size = 5

    # Construye el vocabulario de tu modelo utilizando las palabras del corpus tokenizado
    model = Word2Vec(corpus_tok, vector_size=vector_size, window=window_size, min_count=2, workers=4)

    # Entrena el modelo Word2Vec con el corpus tokenizado
    #model.train(corpus_tok, total_examples=len(corpus_tok), epochs=1000)

    # Guarda el modelo entrenado para su uso posterior
    model.save(name_model)

    return model

# word_to_analyze_companies
def word_to_analyze_companies(Word_to_analize,ASW_model,Globant_model,Accenture_model):

    similaresASW = ASW_model.wv.most_similar(Word_to_analize,topn=5)
    similaresGlobant = Globant_model.wv.most_similar(Word_to_analize,topn=5)
    similaresAccenture = Accenture_model.wv.most_similar(Word_to_analize,topn=5)
    print("----------------------------------------------------------------------------------------")
    print("Word: ", Word_to_analize)
    print("ASW: ", similaresASW)
    print("Globant: ",similaresGlobant)
    print("Accenture: ",similaresAccenture)