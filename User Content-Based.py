#importo il dataset
import time
import pandas as pd
import numpy as np
#importo le librerie per estrarre i dati e lavorarci
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, cosine_distances, rbf_kernel, sigmoid_kernel, pairwise_distances
from surprise import Dataset, Reader, BaselineOnly, KNNBasic, SVD
from surprise import accuracy
from itertools import islice

# SPIEGAZIONE DEL CODDICE
# 1) inserimento di un user id dell'utente da raccomandare
# 2) visualizzo gli item che ha votato e li salvo in una lista
# 3) applico il filtraggio CONTENT-BASED in base al contenuto tramite TfidfVectorizer e linear kernel
# 4) salvo tutti i coefficienti di similarità in una lista unica (all_similarities) formata da un numero di vettori pari
# al numero di item valutate
# 5) chiedo in input il numero di items da raccomandare (provato con 10)
# 6) estraggo da tutte le similarità il numero di item da raccomandare per ogni vettore (nel nostro caso 30)
# 7) ordino tutte le item in base al coefficiente di similarità
# 8) stampo a video le 10 items più simili per coefficiente di similarità

df = pd.read_csv('ratings.csv')
data = pd.read_csv('1662574418893344.csv')

reader = Reader(rating_scale=(1,10))

dataset = Dataset.load_from_df(df[['User_ID', 'Food_ID', 'Rating']], reader)

algo = BaselineOnly()

trainset = dataset.build_full_trainset()
algo.fit(trainset)

testset = trainset.build_testset()

user_ratings = trainset.ur

#stampa delle categorie
print(data['C_Type'].unique())

# inserimento user_id da raccomandare
user_to_recommend = int(input("Inserisci numero utente da raccomandare: "))

# inserimento tipologia di cibo, oppure 0
type_of_food = input("Inserisci una categoria tra quelle specificate sopra, oppure 0 per avere una raccomandazione di default: ")


#controllo se presente nel trainset
if trainset.knows_user(user_to_recommend):
    #print(f"Utente {user_to_recommend} presente nel trainset")
    #devo prelevare i ratings dell'utente da raccomandare
    usr_rat = user_ratings.get(user_to_recommend)
    num_ratings = len(usr_rat)
    print(f"I rating che l'utente {user_to_recommend} ha dato sono in totale {num_ratings} e sono i seguenti:")
    for item_id, ratings in usr_rat:
        print(f"{data.iloc[item_id]['Name']}  --> valutato {ratings}")
else:
    print(f"Utente {user_to_recommend} non presente nel dataset")


print(usr_rat)

print("--------------------------------------")
print("Inizia la raccomandazione in base al contenuto (CONTENT-BASED)")

#recupero tutti gli indici dei cibi valutati dall'utente
# all index to recommend contiene tutti gli item_id degli item che ha valutato l'utente selezionato
all_index_to_recommend = []
for item_id, _ in usr_rat:
   all_index_to_recommend.append(item_id)

print(all_index_to_recommend)

# ora per ogni cibo devo raccomandare tutti i cibi simili per contenuto


#IDEA --> calcolo similarità tra le descrizioni dei cibi votati dall'utente e descrizioni dei cibi estratti/tutti i cibi

# devo verificare se è stato applicato un filtro
# se applicato un filtro allora raccomando cibi simili di quel determinato filtro
# altrimenti calcolo la similarità tra tutti i cibi
description_list = data['Describe'].tolist()
print(description_list)
#programma normale
vectorizer = TfidfVectorizer(stop_words='english')
#trasformo in una tfidf matrix
descr_vector = vectorizer.fit_transform(description_list)


# ritorno i primi n cibi raccomandati
n = int(input("Insersci quanti cibi simili vuoi scoprire:"))

# CONTENT BASED FILTERING
algo = [linear_kernel, cosine_similarity, cosine_distances, rbf_kernel, sigmoid_kernel, pairwise_distances]

for algorithm in algo:

  # per la valutazione del running time
  start = time.time()

  print(f"STO UTILIZZANDO L'ALGORITMO {str(algorithm).split(' ')[1].upper()}")
  name_of_algorithm = str(algorithm).split(' ')[1]

  #ora devo portarla in una forma più compatta per calcolare la similarità tramite il kernel lineare
  matrice_compatta = algorithm(descr_vector, descr_vector)


  #matrice che mi salva tutti i coefficienti
  all_similarities = []


  for i in  all_index_to_recommend:

      # salvo i valori in un vettore
      sim_scores = list(enumerate(matrice_compatta[i]))

      # Filtra gli elementi con similarità diversa da 0.0 e 1.0
      sim_scores_not_zero = [(indice, similarita) for indice, similarita in sim_scores if similarita != 0.0 and similarita != 1.0]


      # li ordino per quelli più simili (stampa numero item, e similarità)
      top_scores = sorted(sim_scores_not_zero, key=lambda x: x[1], reverse = True)

      #elimino il primo elemento in modo tale da escludere l'item uguale a quella che ho scritto
      # nella lista ordinata il primo elemento sarà quello con coefficiente = 1
      top_scores = top_scores[1:]
      all_similarities.append(top_scores)


  print("ALL SIMILARITIES")
  print(all_similarities)
  # all_similarities contiene un numero di vettori pari al numero di items

  # a questo punto vengono stampate tutte le similarità degli item che l'utente ha valutato -> all_similarities

  #-------------------------------------------------------------------
  # devo considerare due casi diversi
  # CASO 1 -> non ho inserito nessuna tipologia e quindi prendo i primi 10 elementi per ogni cibo nella lista dei votati e li inserisco in una lista
  # in seguito ordino la lista ancora in modo descrescente
  # e stampo i primi n elementi

  # CASO 2 -> ho inserito una tipologia e quindi elimino gli items che non hanno quella determinata tipologia
  # salvo gli items con quella tipologia in un dizionario
  # riordino il dizionario
  # stampo le coppie chiave-valore

  # CASO 1 -> non ho inserito nessun tipo di categoria
  if type_of_food == '0':

    dic = {}
    for i in range (0, len(all_similarities)):
      for j in range (0, len(all_similarities[i])):
          dic[data.loc[all_similarities[i][j][0], 'Name']] = all_similarities[i][j][1]

    #ordinamento del dizionario in base all'algoritmo
    if name_of_algorithm == 'cosine_distances' or name_of_algorithm == 'pairwise_distances':
      dic_sorted = dict(sorted(dic.items(), key=lambda item: item[1], reverse=False))
    else:
      dic_sorted = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))


    # Prendi i primi n elementi del dizionario
    dic_sorted = dict(islice(dic_sorted.items(), n))

    #stampo gli elementi
    for key, value in dic_sorted.items():
      print(f"{key} -----> {value * 100}")


  #CASO 2 -> ho inserito una categoria
  else:
    dic = {}
    # se ho inserito la categoria allora qui è il momento di filtrare
    for i in range (0, len(all_similarities)):
      for j in range (0, len(all_similarities[i])):
        #qui estraggo
        if(data.loc[all_similarities[i][j][0], 'C_Type'] == type_of_food):
          dic[data.loc[all_similarities[i][j][0], 'Name']] = all_similarities[i][j][1]


    #ordinamento del dizionario in base all'algoritmo
    if name_of_algorithm == 'cosine_distances' or name_of_algorithm == 'pairwise_distances':
      dic_sorted = dict(sorted(dic.items(), key=lambda item: item[1], reverse=False))
    else:
      dic_sorted = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))


    # Prendi i primi n elementi del dizionario
    dic_sorted = dict(islice(dic_sorted.items(), n))

    #stampo gli elementi
    for key, value in dic_sorted.items():
      #in modo che non consigli cibi che l'utente abbia già votato
      indice_cibo = data.loc[data['Name'] == key].index[0]
      if indice_cibo not in all_index_to_recommend:
        print(f"{key} -----> {value * 100}")

  end = time.time()
  # Calcola il tempo trascorso
  elapsed_time = end - start

  print(f"Tempo di esecuzione: {elapsed_time} secondi")

  print("------------------------------------------------------------------")




