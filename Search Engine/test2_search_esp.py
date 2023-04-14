import pandas as pd
import string
import spacy
import gensim
import numpy as np
from gensim.models import KeyedVectors


## SI TIENEN PROBLEMAS CON SPACY INSTALEN !python -m spacy download es_core_news_sm 

df = pd.read_csv("articles_paragraphs.csv")

df_esp = df[df['language_code'] =='es'].reset_index()

#########################################PROCESS TEXT LEMMA###############################################################



# load spacy nlp model
nlp = spacy.load('es_core_news_sm')

# define function for pre-processing and tokenization
def preprocess_text_lemma(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # lemmatize
    doc = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc]
    # remove stopwords and short words
    stopwords = spacy.lang.es.stop_words.STOP_WORDS
    tokens = [token for token in lemmatized_text if token not in stopwords and len(token) > 2]
    return tokens

# apply pre-processing and tokenization to the 'content' column of each row
tokenized_paragraphs_lemma = []
for paragraph in df_esp['content']:
    tokens = preprocess_text_lemma(paragraph)
    tokenized_paragraphs_lemma.append(tokens)



##########################################  MODELO WORD 2 VEC ###################################################################


# Train Word2Vec model
lemmaModel = gensim.models.Word2Vec(tokenized_paragraphs_lemma,vector_size=40, window=15, min_count=2)

# Calculate the meaning vector per paragraph
paragraph_vectors_lemma = []
for paragraph_tokens in tokenized_paragraphs_lemma:
    vectors = []
    for token in paragraph_tokens:
        if token in lemmaModel.wv.key_to_index:
            vectors.append(lemmaModel.wv[token])
    if len(vectors) > 0:
        paragraph_vectors_lemma.append(np.mean(vectors, axis=0))
    else:
        paragraph_vectors_lemma.append(np.zeros(lemmaModel.vector_size))


##################################### AGREGAR LA VARIABLE DE VECTORES ########################################################


df_esp['vector'] = paragraph_vectors_lemma


################################ FUNCION DE COMPARACION DE COSENO ENTRE LISTA Y VECTOR############################################



def cosine_similarity_list(vectors_list, query_vector):
    #Compute the cosine similarity between the vector representation of the input and the vector representations of each sentence in the text
    similarity_scores = []
    for vector in vectors_list:
        score = query_vector.dot(vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        similarity_scores.append(score)

    # Sort the sentences in descending order of their cosine similarity to the input and return the top-N most similar sentences
    n = 100
    most_similar_sentences = [[vectors_list[idx],idx] for idx in np.argsort(similarity_scores)[::-1][:n] if np.sum(vectors_list[idx]) != 0]

    return most_similar_sentences[:20]

#####################################INPUT DEL MOTOR DE BUSQUEDA#######################################################

userPrompt = "la mayor utilidad de la inteligencia artificial" ######### ACA SE HACE EL INPUT DEL MOTOR DE BUSQUEDA
tokenized_prompt = preprocess_text_lemma(userPrompt)


promptVector_lemma = np.zeros((lemmaModel.vector_size,))
word_count = 0

for token in tokenized_prompt:
    if token in lemmaModel.wv.key_to_index:
        promptVector_lemma += lemmaModel.wv[token]
        word_count += 1
        

if word_count > 0:
    promptVector_lemma /= word_count


##################################COMPARACION DE LOS VECTORES CON EL PROMPT##################################################

var = cosine_similarity_list(df_esp['vector'],promptVector_lemma) ## RETRONA UNA LISTA DE ORDEN DE BUSQUEDA CON var[0][1] SIENDO
# PRIMERA POCISION DE LA BUSQUEDA, Y RETORNANDO SU INDEX EN EL DF DE DF_ENG


###############################                     FIN                       #################################


