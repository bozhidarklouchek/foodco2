"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer, models
from transformers import pipeline
import scipy.spatial
import argparse
import json
import requests
from requests.auth import HTTPBasicAuth
import read_files as read
import os
import csv


# Added food NER model to extract ingredients from instructions
ner_model_path = 'food_ner'
ner = pipeline('ner', model=ner_model_path + '/bert/bert', ignore_labels=[])

ENTITY_LINKING_API = ''
CO2_API = ''

#load labels json
labels_filename = ner_model_path + '/labels.json'
json_file = open(labels_filename, 'r')
json_object = json.loads(json_file.read())
json_file.close()

original_labels_dict = json_object['labels']

# load config file
config_filename = ner_model_path + '/bert/bert/config.json'
json_file = open(config_filename, 'r')
json_object = json.loads(json_file.read())
json_file.close()

label_dict = json_object['label2id']

def get_nes(ner_results, input_text):
  ne_strings = []
  named_entities = []
  current_ne = None
  ne_end = 0
  for result in ner_results:
    entity_label = result['entity']
    token = result['word']
    score = result['score']
    label_str = str(label_dict[entity_label])
    label = original_labels_dict[label_str]
    start_char = result['start']
    end_char = result['end']
    #print(token , label, score, start_char, end_char)
    if 'B-FOOD' in label:
      if current_ne == None:
        current_ne = (start_char, end_char, score)
      elif '##' in token:
        #named_entities.append(current_ne)
        current_ne = (current_ne[0], end_char, score)
    elif label == 'O':
      if '##' in token and current_ne!=None:
        current_ne = (current_ne[0], end_char, score)
      elif current_ne != None:
        named_entities.append(current_ne)
        current_ne = None

    elif 'I-FOOD' in label:
      if current_ne != None:
        current_ne = (current_ne[0], end_char, score)
      else:
        current_ne = (start_char, end_char, score)
  if current_ne != None:
    named_entities.append(current_ne)

  for ne in named_entities:
    ne_strings.append(input_text[ne[0]:ne[1]])

  return ne_strings
  
def api_get(url, ingredient, auth=None):
    url = f'{url}{ingredient}'
    try:
        response = None
        if(auth == None):
            response = requests.get(url)
        else:
            response = requests.get(url, auth=auth)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # If successful, return the response content and decode from bytes to string
            return response.content.decode('UTF-8')
        else:
            # If not successful, print an error message
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except Exception as e:
        # Print any exception that occurred during the request
        print(f"An error occurred: {str(e)}")
        return None

def get_ingredient_info(raw_ingredient_string):
    normalised = api_get(ENTITY_LINKING_API, raw_ingredient_string)
    ghg = json.loads(api_get(CO2_API, normalised, auth=HTTPBasicAuth('username', 'password')))['ghg']
    return {'normalised': normalised, 'ghg': ghg}

   
def get_top_k_results(sentences, query, query_ingredient, results, limit):
    substitutes = []
    
    query_instruction = query[query.find(':') + 1:]
    ner_results = ner(query_instruction)
    query_nes = get_nes(ner_results, query_instruction)
    
    # Remove specified ingredient so set can be compared against suggestions
    query_nes.remove(query_ingredient)
    query_ingred_set = set(query_nes)

    print(f'Replace detected raw string ingredient in query: {query_ingredient}')
    
    query_ingredient_info = get_ingredient_info(query_ingredient)
    print(f"Query normalised raw string ingredient: {query_ingredient_info['normalised']}")
    print(f"Query Co2: {query_ingredient_info['ghg']}")
    
    for idx, score in results:
        suggestion = sentences[idx]
        
        # Remove recipe title from replacement to extract REPLACEMENT_INGREDIENT (right before colon)
        suggestion_instruction = suggestion[suggestion.find(':') + 1:]
        ner_results = ner(suggestion_instruction)
        suggestion_nes = get_nes(ner_results, suggestion_instruction)
        
        suggestion_ingred_set = set(suggestion_nes)
        
        replacements = list(suggestion_ingred_set - query_ingred_set)
        
        if(len(replacements) == 0):
            print(f"Removing suggestion already in instruction: {suggestion_ingredient_info['normalised']}")
            continue
        
        # Take first
        replacement = replacements[0]
        suggestion_ingredient_info = get_ingredient_info(replacement)
        
        # Skip if the same ingredient suggested
        if(suggestion_ingredient_info['normalised'] == query_ingredient_info['normalised']):
            print(f"Removing suggestion with same name: {suggestion_ingredient_info['normalised']}")
            continue
        # Skip if ghg higher
        if(suggestion_ingredient_info['ghg'] >= query_ingredient_info['ghg']):
            print(f"Removing suggestion with >= Co2: {suggestion_ingredient_info['normalised']}, {suggestion_ingredient_info['ghg']}")
            continue
        
        # Else add
        substitutes.append((idx, replacement, score, suggestion_ingredient_info['ghg']))
        if(len(substitutes) >= limit):
            return substitutes
        
        
def main(model_path, model_type,sentence_corpus, query, query_ingredient, ref_ingredient):
    if model_type.lower() in ["bert"]:
        word_embedding_model = models.BERT(model_path)

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        #### load sentence BERT models and generate sentence embeddings ####
    else:
        #### load sentence BERT models and generate sentence embeddings ####
        embedder = SentenceTransformer(model_path)
    corpus_embeddings = read.read_from_pickle(os.path.join(sentence_corpus, "embeddings.pkl"))
    corpus = read.read_from_tsv("data/instructions_to_embed/input.tsv")
    sentences = [item for row in corpus for item in row]

    query_embedding = embedder.encode([query])

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    # closest_n = 5

    distances = scipy.spatial.distance.cdist(query_embedding, corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    
    top_results = get_top_k_results(sentences, query, query_ingredient, results, 5)


    
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    print("\n\n======================\n\n")

    for idx, replacement, distance, ghg in top_results:
        print(f"{sentences[idx].strip()} | Score: {round(1 - distance, 4)} | Co2: {ghg}")
        
    field_names = ['query',
                   'query_ingredient',
                   'ref_ingredient',
                   'pred1_ingredient',
                   'pred2_ingredient',
                   'pred3_ingredient',
                   'pred4_ingredient',
                   'pred5_ingredient']
                   
    output_file = 'data/output/predictions.csv'
    existing_rows = []
    
    if os.path.exists(output_file):
        with open('data/output/predictions.csv', 'r', newline='') as csvfile:
            existing_rows = [r for r in csv.reader(csvfile)][1:]
        
        
    with open('data/output/predictions.csv', 'w', newline='', encoding='utf-8') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(field_names)
    
        for row in existing_rows:
            csv_writer.writerow(row)
        
        csv_writer.writerow([query,
                             query_ingredient,
                             ref_ingredient,
                             top_results[0][1],
                             top_results[1][1],
                             top_results[2][1],
                             top_results[3][1],
                             top_results[4][1]])
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentence embedding for each sentence in the sentence corpus ')

    parser.add_argument('-model',
                        help='the direcotory of the model',required= True)

    parser.add_argument('-model_type',
                        help='the type of the model, sentence_bert or just bert',required= True)

    parser.add_argument('-embeddings',
                        help='the direcotory of the sentence embeddings',required=True)

    parser.add_argument('-query',
                        help='query to comapre embeddings to',required=True)
                        
    parser.add_argument('-query_ingredient',
                        help='raw ingredient in query to replace',required=True)
                        
    parser.add_argument('-ref_ingredient',
                        help='reference ingredient in query to replace',required=True)

    args = parser.parse_args()
    model_path = args.model
    model_type = args.model_type
    corpus_embedding = args.embeddings
    query = args.query
    query_ingredient = args.query_ingredient
    ref_ingredient = args.ref_ingredient

    main(model_path, model_type, corpus_embedding, query, query_ingredient, ref_ingredient)