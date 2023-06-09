import pandas as pd
import spacy
import string
from langdetect import detect
import re 
import nltk
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel

nlp = spacy.load("en_core_web_lg")

contractions_dict = { "ain’t": "are not", "’s":" is", "aren’t": "are not", "can’t": "cannot", 
                     "can’t’ve": "cannot have", "’cause": "because", "could’ve": "could have", 
                     "couldn’t": "could not", "couldn’t've": "could not have", "didn’t": "did not", 
                     "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", 
                     "hadn’t’ve": "had not have", "hasn’t": "has not", "haven’t": "have not",
                     "he’d": "he would", "he’d’ve": "he would have", "he’ll": "he will", 
                     "he’ll’ve": "he will have", "how’d": "how did", "how’d’y": "how do you", 
                     "how’ll": "how will", "i’d": "i would", "i’d’ve": "i would have", "i’ll": "i will",
                     "i’ll’ve": "i will have", "i’m": "i am", "i’ve": "i have", "isn’t": "is not",
                     "it’d": "it would", "it’d’ve": "it would have", "it’ll": "it will", 
                     "it’ll’ve": "it will have", "let’s": "let us", "ma’am": "madam", "mayn’t": "may not",
                     "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have",
                     "must’ve": "must have", "mustn’t": "must not", "mustn’t’ve": "must not have",
                     "needn’t": "need not", "needn’t’ve": "need not have", "o’clock": "of the clock",
                     "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not",
                     "sha’n’t": "shall not", "shan’t’ve": "shall not have", "she’d": "she would",
                     "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", 
                     "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have",
                     "so’ve": "so have", "that’d": "that would", "that’d’ve": "that would have",
                     "there’d": "there would", "there’d’ve": "there would have", "they’d": "they would",
                     "they’d’ve": "they would have","they’ll": "they will", "they’ll’ve": "they will have",
                     "they’re": "they are", "they’ve": "they have", "to’ve": "to have", "wasn’t": "was not",
                     "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have",
                     "we’re": "we are", "we’ve": "we have", "weren’t": "were not","what’ll": "what will",
                     "what’ll've": "what will have", "what’re": "what are", "what’ve": "what have",
                     "when’ve": "when have", "where’d": "where did", "where’ve": "where have", 
                     "who’ll": "who will", "who’ll’ve": "who will have", "who’ve": "who have",
                     "why’ve": "why have", "will’ve": "will have", "won’t": "will not",
                     "won’t’ve": "will not have", "would’ve": "would have", "wouldn’t": "would not",
                     "wouldn’t’ve": "would not have", "y’all": "you all", "y’all’d": "you all would",
                     "y’all’d'’ve": "you all would have", "y’all’re": "you all are",
                     "y’all’ve": "you all have", "you’d": "you would", "you’d’ve": "you would have",
                     "you’ll": "you will", "you’ll’ve": "you will have", "you’re": "you are",
                     "you’ve": "you have"}


def expand_contractions(s, contractions_dict=contractions_dict):

    """
    Replace contractions in a string with their expanded form.

    Args:
    - s: str, input string to expand contractions
    - contractions_dict: dict, a dictionary of contractions and their corresponding expanded form

    Returns:
    - str, the input string with contractions replaced by their expanded form
    """

    contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def clean_hashtag_url(post):

    """
    Remove hashtags and URLs from a string.

    Args:
    - post: str, the input string to clean
    
    Returns:
    - str, the input string with hashtags and URLs removed
    """
    return " ".join(word for word in post.split(' ') if ("#" not in word and "http" not in word))

def punct_space(token):

    """
    Check if a token is a punctuation or a space.

    Args:
    - token: spacy.Token, a token from a spacy document
    
    Returns:
    - bool, True if the token is a punctuation or a space, False otherwise
    """

    return token.is_punct or token.is_space

def rm_pattern(post):

    """
    Remove specific patterns from a string.

    Args:
    - post: str, the input string to remove patterns from
    
    Returns:
    - str, the input string with the specified patterns removed
    """

    post = re.sub("…see more",'', post) 
    post = re.sub('http','',post)
    return post

def preprocess(post):

    """
    Preprocess a string by removing stop words, punctuation, and other unwanted characters.

    Args:
    - post: str, the input string to preprocess
    
    Returns:
    - str, the preprocessed string
    """
    
    clean_text = post.translate(str.maketrans("", "", string.punctuation))
    clean_text = clean_text.replace("\n", " ")
    clean_text = clean_text.replace("\u200d", "")
    clean_text = clean_text.replace("\u200b", "")
    clean_text = clean_text.replace("▪", "")
    clean_text = clean_text.replace("’", "")
    clean_text = clean_text.replace("”", "")
    clean_text = clean_text.replace("we", "")
    clean_text = clean_text.lower()
    names = ['díaz',"catherine","kasper",'cristian',"and", 'perez', "carla","acosta",'we','globant',"equinox","sebastian","juan","juan sebastián","favio","felipe","francisco","angie","jefferson",'1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    stop_words = set(stopwords.words('english')+ names)
    clean_text = " ".join([word for word in clean_text.split() if word not in stop_words])
    clean_text = " ".join(clean_text.split())
    
    return clean_text

def rules(token):

    """
    Define rules for filtering tokens.

    Args:
    - token: spacy.Token, a token from a spacy document
    
    Returns:
    - list of bools, a list of True or False values indicating whether the token passes the filtering rules or not
    """
    return [not punct_space(token)] 

def corpus_cleaning(posts):

    """
    Clean a corpus of posts by applying a set of rules to each token.

    Args:
    - posts: pandas.Series, a series of posts to clean
    
    Yields:
    - str, a cleaned post from the input corpus
    """
    
    for post in nlp.pipe(posts.apply(rm_pattern)):
        yield ' '.join([token.lemma_ for token in post if all(rules(token))])

def main_token(json_name,column_name_corpus):

    """
    Clean and preprocess a corpus of posts, apply bigram model, and return the cleaned corpus.

    Args:
    - json_name: str, the name of the JSON file containing the corpus of posts
    - column_name_corpus: str, the name of the column in the JSON file containing the posts
    
    Returns:
    - list of str, a cleaned and preprocessed corpus of posts
    """

    nltk.download('stopwords')
    df = pd.read_json(json_name)[:261]

    df['language'] = df[column_name_corpus].apply(detect)
    df=df[df['language']!= 'es'].drop('language', axis=1)

    corpus=df[column_name_corpus]

    corpus = corpus.apply(expand_contractions)

    corpus = corpus.apply(clean_hashtag_url)
    corpus = corpus.apply(preprocess)

    preprocessed_posts = corpus_cleaning(corpus)
    streamed_posts = (post.split(' ') for post in preprocessed_posts)
    all_posts = []
    for streamed_post in streamed_posts:
        post = ' '.join(streamed_post)
        all_posts.append(post)

    df['descripcion_clean'] = all_posts

    preprocessed_posts = corpus_cleaning(corpus)
    streamed_posts = (post.split(' ') for post in preprocessed_posts)
    bigram_model = Phrases(streamed_posts,min_count=5,threshold=10)

    bigram_posts = []

    preprocessed_posts = corpus_cleaning(corpus)
    streamed_posts = (post.split(' ') for post in preprocessed_posts)

    for streamed_post in streamed_posts:
        bigram_post = ' '.join(bigram_model[streamed_post])
        bigram_posts.append(bigram_post)

    clean_corpus=bigram_posts

    return clean_corpus