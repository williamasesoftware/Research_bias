def word_to_analyze_companies(Word_to_analize,ASW_model,Globant_model,Accenture_model):

    similaresASW = ASW_model.wv.most_similar(Word_to_analize,topn=5)
    similaresGlobant = Globant_model.wv.most_similar(Word_to_analize,topn=5)
    similaresAccenture = Accenture_model.wv.most_similar(Word_to_analize,topn=5)
    print("----------------------------------------------------------------------------------------")
    print("Word: ", Word_to_analize)
    print("ASW: ", similaresASW)
    print("Globant: ",similaresGlobant)
    print("Accenture: ",similaresAccenture)


def distances_to_target_gender(target, ASW_model, Globant_model, Accenture_model):
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
    ASW_d_1 = ASW_model.wv.distance(target1, target2)
    Globant_d_1 = Globant_model.wv.distance(target1, target2)
    Accenture_d_1 = Accenture_model.wv.distance(target1, target2)

    print(f"\n Distances {target1} with {target2}:\n")
    print("ASW: {:.4f}, Globant: {:.4f}, Accenture: {:.4f}".format(ASW_d_1, Globant_d_1, Accenture_d_1))