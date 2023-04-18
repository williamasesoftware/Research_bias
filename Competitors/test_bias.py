def word_to_analyze_companies(Word_to_analize,ASW_model,Globant_model,Accenture_model):

    """
    Function that receives a word to analyze and three models to obtain the 5 most similar words from each model.

    Parameters:
    - Word_to_analize (str): word to analyze.
    - ASW_model (gensim.models.word2vec.Word2Vec): model containing ASW data.
    - Globant_model (gensim.models.word2vec.Word2Vec): model containing Globant data.
    - Accenture_model (gensim.models.word2vec.Word2Vec): model containing Accenture data.

    Returns:
    - None

    Prints the most similar words to the given word from each model.
    """

    similaresASW = ASW_model.wv.most_similar(Word_to_analize,topn=5)
    similaresGlobant = Globant_model.wv.most_similar(Word_to_analize,topn=5)
    similaresAccenture = Accenture_model.wv.most_similar(Word_to_analize,topn=5)
    print("----------------------------------------------------------------------------------------")
    print("Word: ", Word_to_analize)
    print("ASW: ", similaresASW)
    print("Globant: ",similaresGlobant)
    print("Accenture: ",similaresAccenture)


def distances_to_target_gender(target, ASW_model, Globant_model, Accenture_model):

    """
    Function that receives a target word and three models to calculate the distance from the word to 'man' and 'woman' in each model.

    Parameters:
    - target (str): target word to calculate the distances.
    - ASW_model (gensim.models.word2vec.Word2Vec): model containing ASW data.
    - Globant_model (gensim.models.word2vec.Word2Vec): model containing Globant data.
    - Accenture_model (gensim.models.word2vec.Word2Vec): model containing Accenture data.

    Returns:
    - None

    Prints the distances from the target word to 'man' and 'woman' in each model.
    """

    ASW_d_1=ASW_model.wv.distance("woman",target)
    ASW_d_2=ASW_model.wv.distance("man",target)
    Globant_d_1=Globant_model.wv.distance("woman",target)
    Globant_d_2=Globant_model.wv.distance("mens",target)
    Accenture_d_1=Accenture_model.wv.distance("woman",target)
    Accenture_d_2=Accenture_model.wv.distance("man",target)

    print(f"\n Word: {target}\n")
    print(f"Distances 'woman' with {target}:")
    print("ASW: {:.4f}, Globant: {:.4f}, Accenture: {:.4f}".format(ASW_d_1, Globant_d_1, Accenture_d_1))
    print(f"Distances with 'man' with {target} :")
    print("ASW: {:.4f}, Globant: {:.4f}, Accenture: {:.4f}".format(ASW_d_2, Globant_d_2, Accenture_d_2))


def distances_between_targets(target1, target2, ASW_model, Globant_model, Accenture_model):
    
    """
    Function that receives two target words and three models to calculate the distance between the two words in each model.

    Parameters:
    - target1 (str): first target word.
    - target2 (str): second target word.
    - ASW_model (gensim.models.word2vec.Word2Vec): model containing ASW data.
    - Globant_model (gensim.models.word2vec.Word2Vec): model containing Globant data.
    - Accenture_model (gensim.models.word2vec.Word2Vec): model containing Accenture data.

    Returns:
    - None

    Prints the distance between the two target words in each model.
    """

    ASW_d_1 = ASW_model.wv.distance(target1, target2)
    Globant_d_1 = Globant_model.wv.distance(target1, target2)
    Accenture_d_1 = Accenture_model.wv.distance(target1, target2)

    print(f"\n Distances {target1} with {target2}:\n")
    print("ASW: {:.4f}, Globant: {:.4f}, Accenture: {:.4f}".format(ASW_d_1, Globant_d_1, Accenture_d_1))