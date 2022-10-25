from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_number, remove_itemized_bullet_and_numbering, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, check_spelling, remove_url, remove_email, remove_phone_number, remove_ssn, remove_whitespace
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import csv
import os.path
import json
import string

datafr = pd.read_csv('../data-files/master-ner-results.csv')

# get data (full body texts of documents)
f = open('../data-files/data.json')
data = json.load(f)

# get data (the structured data which Malte processed into lines with metadata)
sf = open('../data-files/studies_on_water_scraped.json')
raw_data = json.load(sf)

nf = open('../data-files/ngram_replacements.json')
ngram_replacements = json.load(nf)

# define function to lookup correct ID for document in studies_on_water_scraped.json
# before this, I was using the INDEX of the document in the JSON array of this file as its ID.
def lookup_correct_docid(old_key):
    global raw_data
    return raw_data[int(old_key)]['meta']['id']

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

# stopwords (niche irrelevant characters)
irrelevant_tokens = ['et', 'al.', 'x', 'pdf', 'yes', 'abbrev','fe',
                            'page', 'pp', 'p', 'er', 'doi', 'b', 'c', 'd', 'e',
                            'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 's',
                            't', 'u', 'v', 'w', 'y', 'z','www', 'com', 'org', 'de', 'dx', 'th', 'ii', 'le']

def preprocess_ner(corpus, stopwords):
    global irrelevant_tokens
    # if no custom stopwords are specified use the one curated
    if stopwords is None:
        stopwords = irrelevant_tokens
        
    # apply text-preprocessing preprocessing pipeline
    preprocess_functions = [remove_whitespace, remove_url, remove_ssn, remove_email, remove_phone_number, remove_number, remove_itemized_bullet_and_numbering]
    preprocessed_text = preprocess_text(corpus, preprocess_functions)
    
    # remove extra whitespace (not detected by text-preprocessing library)
    preprocessed_text = re.sub(' +', ' ', preprocessed_text)
    
    # remove stray 'Foreword' at the start of some docs
    if preprocessed_text.startswith('Foreword'):
        preprocessed_text = preprocessed_text[8:]
        
    # remove stray 'Preface' at the start of some docs        
    if preprocessed_text.startswith('Preface'):
        preprocessed_text = preprocessed_text[7:]

    # remove irrelevant stopwords
    for item in stopwords:
        preprocessed_text = preprocessed_text.replace(' ' + item + '. ', '')
        preprocessed_text = preprocessed_text.replace(' ' + item.upper() + '. ', '')
        preprocessed_text = preprocessed_text.replace(' ' + item + ' ', '')
        preprocessed_text = preprocessed_text.replace(' ' + item.upper() + ' ', '')
 
    # strange tokens and characters to remove
    preprocessed_text = preprocessed_text.replace('±', '')
    preprocessed_text = preprocessed_text.replace('€', '')
    preprocessed_text = preprocessed_text.replace('++', '')
    preprocessed_text = preprocessed_text.replace('n.d.', '')
    preprocessed_text = preprocessed_text.replace('[]', '')
    preprocessed_text = preprocessed_text.replace('()', '')
    preprocessed_text = preprocessed_text.replace('(.)', '')
    preprocessed_text = preprocessed_text.replace(', )', ')')
    preprocessed_text = preprocessed_text.replace('-)', ')')
    preprocessed_text = preprocessed_text.replace('%', '')
    preprocessed_text = preprocessed_text.replace('Figure -', '')
    preprocessed_text = preprocessed_text.replace('Figure ', '')
    preprocessed_text = preprocessed_text.replace('Box ..', '.')
    preprocessed_text = preprocessed_text.replace('Box .', '.')
    preprocessed_text = preprocessed_text.replace('Boxes', '')
    preprocessed_text = preprocessed_text.replace('(Act )', '')
    preprocessed_text = preprocessed_text.replace(', ,', '')
    preprocessed_text = preprocessed_text.replace('Section ', '')
    preprocessed_text = preprocessed_text.replace('Table ', '')
    preprocessed_text = preprocessed_text.replace('. .', '.')
    preprocessed_text = preprocessed_text.replace('. . . . . . .', '')
    preprocessed_text = preprocessed_text.replace('/,.', '')
    preprocessed_text = preprocessed_text.replace(' - ', ' ')
    preprocessed_text = preprocessed_text.replace(',al.', '')
    preprocessed_text = preprocessed_text.replace('\uf0b7', '')
    preprocessed_text = preprocessed_text.replace('Principle .', '')
    preprocessed_text = preprocessed_text.replace('[…]', '')
    preprocessed_text = preprocessed_text.replace('/,', '')
    preprocessed_text = preprocessed_text.replace('..', '')
    preprocessed_text = preprocessed_text.replace(' . ', '. ')
    preprocessed_text = preprocessed_text.replace('(.)', '. ')
    preprocessed_text = preprocessed_text.replace('> .', '')
    preprocessed_text = preprocessed_text.replace('()', '')
    
    preprocessed_text = preprocessed_text.replace('����������������������������������������������������������������������', '')
    preprocessed_text = preprocessed_text.replace('������������', '')
    preprocessed_text = preprocessed_text.replace('�����������', '')
    preprocessed_text = preprocessed_text.replace('����������', '')
    preprocessed_text = preprocessed_text.replace('���������', '')
    preprocessed_text = preprocessed_text.replace('��������', '')
    preprocessed_text = preprocessed_text.replace('�������', '')
    preprocessed_text = preprocessed_text.replace('������', '')
    preprocessed_text = preprocessed_text.replace('�����', '')
    preprocessed_text = preprocessed_text.replace('����', '')
    preprocessed_text = preprocessed_text.replace('���', '')
    preprocessed_text = preprocessed_text.replace('��', '')
    preprocessed_text = preprocessed_text.replace('�', '')
    preprocessed_text = preprocessed_text.replace('.-.', '')
    preprocessed_text = preprocessed_text.replace(',-;', '')
    preprocessed_text = preprocessed_text.replace('( ISBN ---- –', '')
    preprocessed_text = preprocessed_text.replace('----', '')

    return preprocessed_text

def replace_ngrams_with_unigrams_named_entities():
    global data
    global datafr
    global entity_dict
    global ngram_replacements
    
    # copy data
    new_data = {}
    keys =[]
    for key in data:
        # keys.append(key)
        # print(key, ' - ', type(key))
        new_data[str(lookup_correct_docid(key))] = data[key]
        
    # keys.sort()
    # print(keys)
    # replace phrases with unigrams
    entity_dict = {}
    
    # print(len(new_data))
    
    for k in new_data:
        new_data[k] = preprocess_ner(new_data[k], stopwords=None)
        
        # first filter the rows pertaining to the given docid
        docid_datafr = datafr[datafr['docid'] == int(k)]
        # print('frame size: ', len(docid_datafr))
        docid_org_per_datafr = docid_datafr[docid_datafr['entity_type'].isin(['ORG', 'PERSON', 'NORP', 'LOC', 'FAC', 'GPE'])]
        # loop through each entity mention (row) in the dataframe
        docid_org_per_datafr = docid_org_per_datafr.reset_index()  # make sure indexes pair with number of rows
        unique_sentences = pd.unique(docid_org_per_datafr['sentence'])

        for sentence in unique_sentences:
            curr_sent_df = docid_org_per_datafr[docid_org_per_datafr['sentence'] == sentence]
            curr_sent_entities = curr_sent_df['entity'].tolist()
            replace_patterns = {}
            for entity in curr_sent_entities:
                if isinstance(entity, str):
                    if entity[0:4] == 'the ':
                        entity = entity[4:]
                    named_entity_tokens = entity.strip().replace('"', '').replace("'", '').replace(",",'').replace('’','').replace('‘','').replace('“','').replace('”','').split()
                    if (len(named_entity_tokens) > 1):
                        # form single token from multiple ones
                        single_token_entity = '_'.join(named_entity_tokens)
                        replace_patterns[entity] = single_token_entity
                        entity_dict[entity] = single_token_entity
                        # print('entity: ', entity, ' - ', 'replacement: ', entity_dict[entity])

            new_sentence = replace_all(sentence, replace_patterns)
            new_data[k] = new_data[k].replace(sentence, new_sentence)
            
    # print('entity_dict size: ', len(entity_dict))

    with open('../data-files/entity_dict.json', 'w') as fp:
        json.dump(entity_dict, fp)

    # Declare a list that is to be converted into a column
    single_tokens = []

    for item in datafr['entity'].tolist():
        if isinstance(item, str):
            if item[:4] == 'the ':
                item = item[4:]

            named_entity_tokens = item.strip().replace('"','').replace("'",'').replace(",",'').replace('’','').replace('‘','').replace('“','').replace('”','').split()
            
            if (len(named_entity_tokens) > 1):
                single_tokens.append(entity_dict[item])
            elif (len(named_entity_tokens) == 1):
                single_tokens.append(item)
            else:
                single_tokens.append(None)
        else:
            single_tokens.append(None)
                          
    datafr['entity_as_single_token'] = single_tokens

    datafr.to_csv('../data-files/master-ner-results-singletokens.csv', index=False)

    return new_data
    
def replace_ngrams_with_unigrams_curated_phrases():
    global ngram_replacements
    
    data = replace_ngrams_with_unigrams_named_entities()
    
    processed_data = {}
    for key in data:
        processed_data[key] = replace_all(data[key], ngram_replacements)       
    
    with open('../data-files/processed_ngram_ner_data.json', 'w') as fp:
        json.dump(processed_data, fp)
        
    return processed_data

def preprocess_word2vec(doctext, custom_stopwords):
    output_sentences = []
    # if no stopwords specified, use NLTKs
    if (custom_stopwords is None):
        custom_stopwords = stopwords.words('english')
        
    # split corpus into sentences
    sentences = sent_tokenize(doctext)
    for sentence in sentences:        
        # split sentence into tokens (words)
        curr_tokens = word_tokenize(sentence)
        # lower case tokens
        for i in range(len(curr_tokens)):
            curr_tokens[i] = curr_tokens[i].lower()

        # remove punctuation and whitespace characters
        tokens_minus_punctuation = []
        for token in curr_tokens:
            token_contains_punct = False
            for chara in token:
                # underscores, dashes and slashes are allowed
                if ((chara in string.punctuation and (chara not in ['_', '-', '/', '\\'])) or (chara in ['©','θ','•','','��'])):
                    token_contains_punct = True
                    break
            if not token_contains_punct:
                test_whitespace = token
                # remove quotes and normalise dashes and slashes
                test_whitespace = test_whitespace.replace('”','')
                test_whitespace = test_whitespace.replace('“','')
                test_whitespace = test_whitespace.replace('‘','')
                test_whitespace = test_whitespace.replace('’','')
                test_whitespace = test_whitespace.replace('-','_')
                test_whitespace = test_whitespace.replace('/','_')
                test_whitespace = test_whitespace.replace('\\','_')
                if (len(test_whitespace) > 0):
                    token = token.strip()
                    tokens_minus_punctuation.append(token)

        # remove stopwords
        cleaned_tokens = [w for w in tokens_minus_punctuation if not w in custom_stopwords]
        
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(w) for w in cleaned_tokens]
        output_sentences.append(lemmatized_tokens)
            
    return output_sentences

