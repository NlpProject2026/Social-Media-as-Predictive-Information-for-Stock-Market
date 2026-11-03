from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


# Loading of english language model without parser & NER 
# (useless for Bow & TF-IDF cause they lose the words' order in documents)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Tokenisation & lemmatisation (without cashtag gestion)
def spacy_tokenizer(text):
    doc = nlp(text)

    # We keep lemma, and drop punctuation & space cases
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]

    return tokens

# Tokenisation & lemmatisation (with cashtag gestion)
def spacy_cashtag_tokenizer(text, ignore_cashtags=False):
    doc = nlp(text)
    tokens = []
    skip_next = False
    
    for i, token in enumerate(doc):
        if skip_next:
            skip_next = False
            continue

        # We deal with cashtag (ex "$AAPL")
        if token.text == '$' and i + 1 < len(doc) and doc[i+1].text.isalnum():
            if ignore_cashtags==False:
                # We merge the "$" & "X" (ex: "$aapl")
                cashtag = token.text + doc[i+1].text
                tokens.append(cashtag.lower())
            skip_next = True # to skip the next token

        # We keep lemma, and drop punctuation & space cases
        elif not token.is_punct and not token.is_space and not token.is_stop:
            tokens.append(token.lemma_.lower())
            
    return tokens

def bag_of_Word(df, column, verbose=False):
    vectorizer_bow = CountVectorizer(tokenizer=spacy_cashtag_tokenizer, token_pattern=None)
    Bow = vectorizer_bow.fit_transform(df[column]).toarray()
    voc_bow = vectorizer_bow.vocabulary_
    if verbose:
        print('Bag of Word matrix :\n', Bow)
        print('Bag of Word matrix shape :', Bow.shape)
        print('Bag of Word matrix min :', Bow.min(), 'Bag of Word matrix max :', Bow.max())
        print('vocabulary :', voc_bow)

    return Bow, voc_bow

def tf_idf(df, column, ignore_cashtags=True, sparse_representation=True, verbose=False):
    vectorizer_tfidf = TfidfVectorizer(tokenizer=lambda text: spacy_cashtag_tokenizer(text, ignore_cashtags), token_pattern=None)
    Tfidf = vectorizer_tfidf.fit_transform(df[column])
    if sparse_representation==False:
        Tfidf = Tfidf.toarray()
    voc_tfidf = vectorizer_tfidf.vocabulary_
    features_tfidf = vectorizer_tfidf.get_feature_names_out()
    if verbose:
        print('TF-IDF matrix :\n', Tfidf)
        print('TF-IDF matrix shape :', Tfidf.shape)
        print('TF-IDF matrix min :', Tfidf.min(), 'Bag of Word matrix max :', Tfidf.max())
        print('vocabulary :', voc_tfidf)

    return Tfidf, voc_tfidf, features_tfidf

def rev_voc(voc):
    rev_voc = {index: word for word, index in voc.items()}

    return rev_voc