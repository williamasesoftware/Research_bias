import pandas as pd
import string
import spacy
import gensim
import numpy as np
from gensim.models import KeyedVectors

df_lemma = pd.read_csv("articles_paragraphs.csv")

nlp = spacy.load('en_core_web_sm')


################################################################################################


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
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    tokens = [token for token in lemmatized_text if token not in stopwords and len(token) > 2]
    return tokens

# apply pre-processing and tokenization to the 'content' column of each row
tokenized_paragraphs_lemma = []
for paragraph in df['content']:
    tokens = preprocess_text_lemma(paragraph)
    tokenized_paragraphs_lemma.append(tokens)


####################################################################################################


# Train Word2Vec model
paragraphModel = gensim.models.Word2Vec(tokenized_paragraphs_lemma, window=5, min_count=1, workers=4)
paragraphModel.save("paragraphModel")

# Calculate the meaning vector per paragraph
paragraph_vectors_lemma = []
for paragraph_tokens in tokenized_paragraphs_lemma:
    vectors = []
    for token in paragraph_tokens:
        if token in paragraphModel.wv.key_to_index:
            vectors.append(paragraphModel.wv[token])
    if len(vectors) > 0:
        paragraph_vectors_lemma.append(np.mean(vectors, axis=0))
    else:
        paragraph_vectors_lemma.append(np.zeros(model.vector_size))



#######################################################################################################


df_lemma['vector'] = paragraph_vectors_lemma


#######################################################################################################

def cosine_similarity_list(vectors_list, query_vector):
    #Compute the cosine similarity between the vector representation of the input and the vector representations of each sentence in the text
    similarity_scores = []
    for vector in vectors_list:
        score = query_vector.dot(vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        similarity_scores.append(score)

    # Sort the sentences in descending order of their cosine similarity to the input and return the top-N most similar sentences
    n = 20
    most_similar_sentences = [[vectors_list[idx],idx] for idx in np.argsort(similarity_scores)[::-1][:n] if np.sum(vectors_list[idx]) != 0]

    return most_similar_sentences[:2]


##############################################################################################################


userPrompt = "medicine using artificial intelligence"  # ACA ES EL INPUT DE BUSQUEDAA

def preprocess_text_lemma(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # lemmatize
    doc = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc]
    # remove stopwords and short words
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    tokens = [token for token in lemmatized_text if token not in stopwords and len(token) > 2]
    return tokens

tokenized_prompt = preprocess_text_lemma(userPrompt)




#################################################################################################

# Preprocesa y tokeniza el input y embedding y devuelve vector

promptVector = np.zeros((paragraphModel.vector_size,))
word_count = 0

for token in tokenized_prompt:
    if token in paragraphModel.wv.key_to_index:
        promptVector += paragraphModel.wv[token]
        word_count += 1

if word_count > 0:
    promptVector /= word_count
    

######################################################################################################3

# en var se guarda el index de los 2 parrafos mas probables

var=cosine_similarity_list(df_lemma['vector'],promptVector)  ## Es esta variable se guardan dos posibles respuestas var[1][1] o var[0][1]

df["content"][var[0][1]]  ## Esto es el parrafo 