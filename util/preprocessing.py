from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_number, remove_itemized_bullet_and_numbering, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, check_spelling, remove_url, remove_email, remove_phone_number, remove_ssn, remove_whitespace
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import os.path
import json
import string
from pathlib import Path

path = Path(os.getcwd())
data_dir = os.path.join(path.parents[0], "data-files")

# get data (the structured data which Malte processed into lines with metadata)
with open(os.path.join(data_dir, "studies_on_water_scraped.json")) as sf:
    raw_data = json.load(sf)

# with open(os.path.join(data_dir, "ngram_replacements.json")) as nf:
#     ngram_replacements = json.load(nf)

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
    
    # these are anomalous text or headings which appear at the start of each document. here we remove them. 
    preprocessed_text = preprocessed_text.replace("format title author subject keywords creator producer creationDate modDate trapped encryption id Preface", "")
    preprocessed_text = preprocessed_text.replace("format title author subject keywords creator producer creationDate modDate trapped encryption id", "")

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
        preprocessed_text = preprocessed_text.replace(' ' + item + '. ', ' ')
        preprocessed_text = preprocessed_text.replace(' ' + item.upper() + '. ', ' ')
        preprocessed_text = preprocessed_text.replace(' ' + item + ' ', ' ')
        preprocessed_text = preprocessed_text.replace(' ' + item.upper() + ' ', ' ')
 
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
    preprocessed_text = preprocessed_text.replace('Figure ', ' ')
    preprocessed_text = preprocessed_text.replace('Box ..', '.')
    preprocessed_text = preprocessed_text.replace('Box .', '.')
    preprocessed_text = preprocessed_text.replace('Boxes', '')
    preprocessed_text = preprocessed_text.replace('(Act )', '')
    preprocessed_text = preprocessed_text.replace(', ,', '')
    preprocessed_text = preprocessed_text.replace('Section ', ' ')
    preprocessed_text = preprocessed_text.replace('Table ', ' ')
    preprocessed_text = preprocessed_text.replace('. .', '.')
    preprocessed_text = preprocessed_text.replace('. . . . . . .', '')
    preprocessed_text = preprocessed_text.replace('/,.', '')
    preprocessed_text = preprocessed_text.replace(' - ', ' ')
    preprocessed_text = preprocessed_text.replace(',al.', '')
    preprocessed_text = preprocessed_text.replace('\uf0b7', '')
    preprocessed_text = preprocessed_text.replace('Principle .', ' ')
    preprocessed_text = preprocessed_text.replace('[…]', '')
    preprocessed_text = preprocessed_text.replace('/,', '')
    preprocessed_text = preprocessed_text.replace('..', '')
    preprocessed_text = preprocessed_text.replace(' . ', '. ')
    preprocessed_text = preprocessed_text.replace('(.)', '. ')
    preprocessed_text = preprocessed_text.replace('> .', ' ')
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

def get_underscore_version(stri):
    result = remove_punctuation_and_lowercase(stri)
    entity_tokens = result.split()
    
    if (len(entity_tokens) > 1):
        single_token_entity = '_'.join(entity_tokens)
        result = single_token_entity
        
    return result

def remove_punctuation_and_lowercase(stri):
    punct = '''!"#$%&'()*+,./:;<=>?@[\]^`{|}~'''
    result = stri.translate(str.maketrans('', '', punct))
    result = result.strip()
    return result.lower()

def replace_ngrams_with_unigrams_named_entities(source_file, ne_results_file, abbreviations_file, output_processed_nerresults_file):
    # 1. load abbreviations file
    with open(os.path.join(data_dir, abbreviations_file)) as f:
        abbr = json.load(f)
        
    # 2. load source text files
    with open(os.path.join(data_dir, source_file)) as f:
        data = json.load(f)

    # 3. load NER results file
    df = pd.read_csv(os.path.join(data_dir, ne_results_file))
    
    # 4. copy data (ensuring we don't overwrite the data in case we need it later)
    new_data = {}
    keys =[]
    for key in data:
        new_data[str(lookup_correct_docid(key))] = data[key]
    
    # 5. do replacements of ngram named entities with single token versions IN THE SOURCE TEXT FILES themselves
    entity_dict = {}
    for k in new_data:
        new_data[k] = preprocess_ner(new_data[k], stopwords=None)
        
        # first filter the rows pertaining to the given docid
        docid_datafr = df[df['docid'] == int(k)]
        docid_org_per_datafr = docid_datafr[docid_datafr['entity_type'].isin(['ORG', 'PERSON', 'NORP', 'LOC', 'FAC', 'GPE'])]
        # loop through each entity mention (row) in the dataframe
        docid_org_per_datafr = docid_org_per_datafr.reset_index()  # make sure indexes pair with number of rows
        unique_sentences = pd.unique(docid_org_per_datafr['sentence'])

        for sentence in unique_sentences:
            curr_sent_df = docid_org_per_datafr[docid_org_per_datafr['sentence'] == sentence]
            curr_sent_entities = curr_sent_df['entity'].tolist()
            normal_replace_patterns = {}
            abbr_replace_patterns = {}
            for entity in curr_sent_entities:
                if isinstance(entity, str):
                    if entity[0:4] == 'the ':
                        entity = entity[4:]
                        
                    entity_minus_punct = remove_punctuation_and_lowercase(entity)

                    if (entity_minus_punct in abbr.keys()):
                        abbr_replace_patterns[entity.strip()] = get_underscore_version(abbr[entity_minus_punct])
                        entity_dict[entity.strip()] = get_underscore_version(abbr[entity_minus_punct])
                    else:
                        normal_replace_patterns[entity_minus_punct] = get_underscore_version(entity.strip())
                        entity_dict[entity.strip()] = get_underscore_version(entity.strip())

            new_sentence = replace_all(sentence, abbr_replace_patterns)
            new_sentence = replace_all(new_sentence.lower(), normal_replace_patterns)
            new_data[k] = new_data[k].replace(sentence, new_sentence.lower())
            
        for item in abbr:
            new_data[k] = new_data[k].replace(' ' + item + ' ', ' ' + get_underscore_version(abbr[item]) + ' ')
            new_data[k] = new_data[k].replace('(' + item + ')', ' ' + get_underscore_version(abbr[item]) + ' ')
            new_data[k] = new_data[k].replace(' ' + item.upper() + ' ', ' ' + get_underscore_version(abbr[item]) + ' ')
            new_data[k] = new_data[k].replace('(' + item.upper() + ')', ' ' + get_underscore_version(abbr[item]) + ' ')

    # 6. save the list of replacements to file (to see them later for analysis or proofreading)
    with open(os.path.join(data_dir, "entity_dict.json"), 'w') as fp:
        json.dump(entity_dict, fp)

    # 7. add a column to the NER RESULTS FILE itself with the single token versions of ngram named entities
    single_tokens = []
    for item in df['entity'].tolist():
        if isinstance(item, str):
            if item[:4] == 'the ':
                item = item[4:]

            named_entity_tokens = remove_punctuation_and_lowercase(item).split()
                                                
            if (len(named_entity_tokens) > 1):
                single_tokens.append(entity_dict[item.strip()])
            elif (len(named_entity_tokens) == 1):
                single_tokens.append(remove_punctuation_and_lowercase(item))
            else:
                single_tokens.append(None)
        else:
            single_tokens.append(None)
                          
    df['entity_as_single_token'] = single_tokens
    
    # 8. clean the NER results file to make it ready for analysis:
    # - remove unnecessary punctation
    # - remove false positive NEs (especially from ORGs)
    # - remove NaN or NULL value rows
    
    # a. remove NaN values
    df = df.dropna()
    
    # b. remove unnecessary punctuation in entities
    df['entity'] = df['entity'].str.replace('“','')
    df['entity'] = df['entity'].str.replace('”','')
    df['entity'] = df['entity'].str.replace('‘','')
    df['entity'] = df['entity'].str.replace('’','')
    df['entity'] = df['entity'].str.replace('()','', regex=False)
    df['entity'] = df['entity'].str.replace('–-','')
    df['entity'] = df['entity'].str.strip() # remove spaces to the left and right of the NE
    df['entity'] = df['entity'].str.lower() # lowercase entities to reduce duplicates according to case

    # c. remove redundant 'the' at the start of named entities
    containingThe = df.loc[df['entity'].str.startswith('the', na=False)] # entities starting with 'the'
    notContainingThe = df.loc[~df['entity'].str.startswith('the', na=False)] # not starting with 'the'
    containingThe['entity'] = containingThe['entity'].apply(lambda x: "" + x[3:])
    # concatenate the cleaned dataframe (without the remove 'the' words) with the one NOT containing 'the'
    df = pd.concat([containingThe, notContainingThe], axis=0)
    
    # d. split the dataframe according to NE types:
    fac_df = df[df['entity_type'] == 'FAC']
    gpe_df = df[df['entity_type'] == 'GPE']
    loc_df = df[df['entity_type'] == 'LOC']
    norp_df = df[df['entity_type'] == 'NORP']
    org_df = df[df['entity_type'] == 'ORG']
    per_df = df[df['entity_type'] == 'PERSON']
    # non organisation type entities in one dataframe
    nonorg_df = pd.concat([fac_df, gpe_df, loc_df, norp_df, per_df], ignore_index=True)
    
    # e. load list of false positive organisations
    fp_orgs_file = open(os.path.join(data_dir,'nonorgs_.txt'), 'r')
    fp_orgs = fp_orgs_file.readlines()
    fp_orgs_set = set()
    for fp_org in fp_orgs:
        fp_org = fp_org.replace('\n', '')
        fp_org = fp_org.replace('\r\n', '')
        fp_org = fp_org.strip()
        fp_orgs_set.add(fp_org)
    
    # f. remove punctuation at the END of a NE string (last character)
    org_df['entity'] = org_df['entity'].str.strip()
    org_df['entity'] = org_df['entity'].str.lower()
    punct_mask = (org_df['entity'].str[-1].isin( 
                  ['!','"','#','$','%','&','\',''','*','+',',','-','.','/',':'
                   ,';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']))

    not_punct_mask = (~org_df['entity'].str[-1].isin( 
                  ['!','"','#','$','%','&','\',''','*','+',',','-','.','/',':'
                   ,';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']))

    punct_df = org_df.loc[punct_mask]
    not_punct_df = org_df.loc[not_punct_mask]
    punct_df['entity'] = punct_df['entity'].str[:-1]
    punct_df['entity'] = punct_df['entity'].str.strip()
    new_df = pd.concat([punct_df, not_punct_df], ignore_index=True)

    # g. i) remove false positive organisations from the dataframe
    unwanted_orgs = list(fp_orgs_set)
    unwanted_orgs.extend(nonorg_df['entity'].tolist())
    cleanorg_df = new_df[~new_df['entity'].isin(unwanted_orgs)]
    
    # g. ii) NEs that end with these words are likely not organisations (further false positives)
    cleanorg_df = cleanorg_df[~cleanorg_df['entity'].str.endswith('system')]
    cleanorg_df = cleanorg_df[~cleanorg_df['entity'].str.endswith('database')]
    cleanorg_df = cleanorg_df[~cleanorg_df['entity'].str.endswith(' plan')]
    cleanorg_df = cleanorg_df[~cleanorg_df['entity'].str.endswith(' strategy')]

    # h. only keep organisations that are mentioned more than once
    cleanorg_df = cleanorg_df[cleanorg_df.groupby('entity').entity.transform(len) > 1]
    nonorg_df = nonorg_df[nonorg_df.groupby('entity').entity.transform(len) > 1]
    
    # i. join up the organisations and non-organisation NE dataframes
    full_ner_dataset_df = pd.concat([cleanorg_df, nonorg_df], ignore_index=True)

    # j. write the NER results to file
    full_ner_dataset_df.to_csv(os.path.join(data_dir, output_processed_nerresults_file), index=False)

    return new_data
    
def replace_ngrams_with_unigrams_curated_phrases(source_texts_file, abbreviations_file, ne_results_file, other_replacements_file, output_processed_text_file, output_processed_nerresults_file):
    # 1. add a new column to the NER results file with underscore single token versions of each named entity
    # 2. replace occurrences of each named entity in the source text files with the corresponding single token version
    # 3. also replace some occurrences of acronyms / abbreviation named entities with full names (from manual curated abbreviations dictionary)
    data = replace_ngrams_with_unigrams_named_entities(source_texts_file, ne_results_file, abbreviations_file, output_processed_nerresults_file)
    
    # 4. load another dictionary of ngram terms that could represent single token concepts (e.g. "water security" becomes "water_security")
    with open(os.path.join(data_dir, other_replacements_file)) as nf:
        other_replacements = json.load(nf)
        
    # 5. do the replacements of ngram concepts (loaded in Step 4) with single token versions in the source texts
    processed_data = {}
    for key in data:
        processed_data[key] = replace_all(data[key], other_replacements)       
    
    # 6. write the processed text to file
    with open(os.path.join(data_dir, output_processed_text_file), 'w') as fp:
        json.dump(processed_data, fp)

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
            cleaned_token = remove_punctuation_and_lowercase(token)
            if (len(cleaned_token) > 0):
                tokens_minus_punctuation.append(cleaned_token)

        # remove stopwords
        cleaned_tokens = [w for w in tokens_minus_punctuation if not w in custom_stopwords]
        
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(w) for w in cleaned_tokens]

        output_sentences.append(lemmatized_tokens)
            
    return output_sentences

