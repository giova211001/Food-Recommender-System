# programma che raccomanda in base al contenuto di parole chiave inserite in input
# controlla la similarità tra parole chiave e descrizione di ogni cibo

import time
import pandas as pd
from translate import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, cosine_distances, rbf_kernel, sigmoid_kernel, pairwise_distances
import numpy as np
from itertools import islice

def aggiungi_al_dizionario(dizionario, chiave, valore):
    # Se la chiave è già presente nel dizionario
    if chiave in dizionario:
        # Mantieni solo il valore più alto
        dizionario[chiave] = max(dizionario[chiave], valore)
    else:
        # Altrimenti, aggiungi la nuova chiave-valore
        dizionario[chiave] = valore


def aggiungi_al_dizionario_cosine(dizionario, chiave, valore):
    # Se la chiave è già presente nel dizionario
    if chiave in dizionario:
        # Mantieni solo il valore più basso
        dizionario[chiave] = min(dizionario[chiave], valore)
    else:
        # Altrimenti, aggiungi la nuova chiave-valore
        dizionario[chiave] = valore


data = pd.read_csv('1662574418893344.csv')

#variabile per capire se utente ha intolleranze
intollerance = None

print("Inserisci un insieme di parole chiave, 0 per terminare l'inserimento:")
# inserimento funzionalità in più -> parole che non devono contenere i cibi!!!
# esempio persona celiaca -> non deve esserci pane, farina
keywords = []
while True:
    input_usr = input()
    if input_usr == '0':
        break #esci dal while
    else:
        keywords.append(input_usr)

print("Le parole chiave DA TENERE IN CONSIDERAZIONE sono le seguenti:")
print(keywords)

print("Se sei intollerante a qualche cibo inseriscilo qui, 0 per terminare")
not_present = []
while True:
  input_usr = input()
  if input_usr == '0':
    break
  else:
    not_present.append(input_usr)

# modifico il valore di intollerance in base alla dimensione di not_present
if len(not_present) == 0:
  #non ho alcuna intolleranza
  intollerance = 0
else:
  #ho intolleranze
  intollerance = 1


translator = Translator(to_lang="en", from_lang="it")

#traduzione parole da tenere in considerazione
keywords_en = []
for i in range (0, len(keywords)):
    translation = translator.translate(keywords[i])
    keywords_en.append(translation)

print(keywords_en)

if intollerance == 1:

  #traduzione parole da NON tenere in considerazione
  not_present_en = []
  for i in range (0, len(not_present)):
    translation = translator.translate(not_present[i])
    not_present_en.append(translation)

  print(not_present_en)

  #ELIMINO DAL DATASET TUTTI I CIBI CHE HANNO PRESENTE NEL NOME O NELLA DESCRIZIONE LE PAROLE DA NON CONSIDERARE
  # Inizializza una lista vuota per conservare i cibi rimossi
  removed_foods = []

  # Applica la rimozione delle righe e conserva i cibi rimossi nella lista
  for word in not_present_en:
    removed_rows = data[data['Name'].str.contains(word, case=False) | data['Describe'].str.contains(word, case=False)]
    removed_foods.extend(removed_rows['Name'].tolist())
    data = data[~data['Name'].str.contains(word, case=False) & ~data['Describe'].str.contains(word, case=False)]

  print(f"Il dataframe adesso ha in totale {len(data)} cibi")

  # Mostra i cibi rimossi
  print("\nCibi rimossi:")
  print(removed_foods)

  print(f"Ho rimosso in totale {len(removed_foods)} cibi dal dataset in base alle parole fornite")

else:
  print("Fortunatamente non hai intolleranze!!! Non ho rimosso alcun cibo dal dataset")


#devo salvare in una lista tutte le descrizioni, nome del cibo di solito è presente nella descrizione
description_list = data['Describe'].tolist()
print(description_list)


#ora devo confrontare la lista di descrizioni con le parole chiave
vectorizer = TfidfVectorizer(stop_words='english')
description_vector = vectorizer.fit_transform(description_list)
keywords_vector = vectorizer.transform(keywords_en)

print("Quanti cibi vuoi che il software ti consigli?")
n = int(input())

#in questo punto dovrei provare diversi tipi di algoritmi con un ciclo for

#COSINE_DISTANCES possibili valori
# -> 0 = vettori sono perfettamente simili
# -> 1 = vettori sono meno simili
# -> 2 = vettori completamente dissimili

algo = [linear_kernel, cosine_similarity, cosine_distances, rbf_kernel, sigmoid_kernel, pairwise_distances]

dict_list = {}
for algorithm in algo:
  
  # per valutare il running time
  start = time.time()
  
  
  print(f"STO UTILIZZANDO L'ALGORITMO {str(algorithm).split(' ')[1].upper()}")
  name_of_algorithm = str(algorithm).split(' ')[1]

  #viene calcolato il coefficiente di similarità tra vettore di descrizioni di tutti i cibi e vettore di parole inserite dall'utente
  cosine_sim = algorithm(description_vector, keywords_vector)

  #ottengo i valori diversi da zero
  nonzero_rows, nonzero_cols = np.nonzero(cosine_sim)


  #ottengo tutti i valori diversi da zero
  cosine_sim_not_zero = cosine_sim[nonzero_rows, nonzero_cols]

  #qui provo a creare il dizionario
  dictionary = {}
  for riga, colonna, valore in zip(nonzero_rows, nonzero_cols, cosine_sim_not_zero):
    #inserisco coppia chiave-valore
    if name_of_algorithm == 'cosine_distances' or name_of_algorithm == 'pairwise_distances':
      #devo salvare il minimo
      aggiungi_al_dizionario_cosine(dictionary, data.loc[data['Describe'] == description_list[riga], 'Name'].values[0], valore)
    else:
      #salvo il massimo
      aggiungi_al_dizionario(dictionary, data.loc[data['Describe'] == description_list[riga], 'Name'].values[0], valore)

  #stampo dizionario che ho creato
  print(dictionary)
  print(f"Il dizionario ha {len(dictionary)} elementi")

  print("ORDINAMENTO DEL DIZIONARIO")
  if name_of_algorithm == 'cosine_distances' or name_of_algorithm == 'pairwise_distances':
    # ordinamento crescente
    dict_sorted = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=False))
  else:
    # ordinamento decrescente
    dict_sorted = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

  #ELIMINO I DUPLICATI
  #salvo le chiavi processate
  chiavi_processate = {}

  for key, value in dict_sorted.items():
    if key not in chiavi_processate:
        chiavi_processate[key] = value * 100

  # Ora chiavi_processate contiene solo le chiavi uniche con i valori moltiplicati per 100
  print(chiavi_processate)

  # Prendi i primi n elementi del dizionario
  dict_sorted = dict(islice(chiavi_processate.items(), n))
  print(dict_sorted)

  print(f"Dimensione del dizionario = {len(dict_sorted)}")

  # 3 casi
  if len(dict_sorted) == n:
    print(f"I {n} cibi più simili in ordine sono i seguenti: ")
    for key, value in dict_sorted.items():
      print(f"{key} con coefficiente di similarità =  {value} ")
  elif len(dict_sorted) < n and len(dict_sorted) != 0:
    print(f"Non sono riuscito a trovare il numero di cibi simili che mi hai richiesto, ma solo {len(dict_sorted)}")
    for key, value in dict_sorted.items():
      print(f"{key} con coefficiente di similarità =  {value} ")
  else:
    print("Non ho trovato cibi corrispondenti")

  end = time.time()
  # Calcola il tempo trascorso
  elapsed_time = end - start

  print(f"Tempo di esecuzione: {elapsed_time} secondi")  
  
  print("------------------------------------------------------------------")


