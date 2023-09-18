from __future__ import print_function
from PyPDF2 import PdfReader
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from nltk.translate.bleu_score import sentence_bleu
from typing import List, Dict
import re
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet


class PDFReader():

    global pdf_dictionary 

    def __init__(self,directory :str) -> None:
        self.directory = directory
        self.pdfs_names = [pdf for pdf in os.listdir(directory)]
        self.pdf_dictionary = {}
        self.infos = {}
    
    def read_text(self) -> str:
        for j in range(len(self.pdfs_names)):
            with open(self.directory + "/" +self.pdfs_names[j], 'rb') as file:
                ## Create a PDF reader object
                reader = PdfReader(file)
                self.infos[j] = reader.metadata
                all_text=""
                ## Get the number of pages in the PDF file
                num_pages = len(reader.pages)
                for i in range(num_pages):
                    ## Get the page object
                    page =  reader.pages[i]
                    ## Extract the text from the page
                    text= page.extract_text()
                    all_text+=text
                self.pdf_dictionary[self.pdfs_names[j]]= all_text
        return self.pdf_dictionary,self.infos
        

class TextPrecessor():
    
    def __init__(self) -> None:
        self.stemming = PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.data = pd.DataFrame({
            "text": [],
            "titre": [],
            "tokenize_text":[],
            "Author" : [],
            "subject" : [],
            "Id" : [],
            "Creator" : [],
            "title" : [],
            })
        self.preprocess_data()


    def tokenize_and_stem(self,text) -> list:
        # Call the read_text method to get the dictionary of PDF content
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.isalpha()]
        tokens=[w for w in tokens if (not w in self.stopwords)&(len(w)>1) ]
        filtered_tokens = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [self.stemming.stem(t) for t in filtered_tokens]
        return stems

    def preprocess_data(self):
        # Call the tokenize_and_stem method to tokenize and stem the PDF content
        pdf_reader = PDFReader("files")
        pdf_dictionary,infos = pdf_reader.read_text()
        for i, pdf_name in enumerate(pdf_dictionary.keys()):
            
            # Append the data to the DataFrame
            self.data = self.data.append({
                "text": pdf_dictionary[pdf_name],
                "titre": pdf_name,
                "tokenize_text": self.tokenize_and_stem(pdf_dictionary[pdf_name]),
                "Author" : infos[i].author ,
                "subject" : infos[i].subject,
                "Id" : i+1,
                "Creator" : infos[i].creator,
                "title" : infos[i].title,
            }, ignore_index=True)
        self.data.fillna("Unkown", inplace=True)
        self.data.replace({"": "Unkown", None: "Unkown"}, inplace=True)
        

class TextVectorizer(TextPrecessor):
    def __init__(self, max_features: int = 200, ngram_range: tuple = (1, 3)) -> None:
        self.instance = TextPrecessor()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stemming = PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features,
                                        stop_words="english",
                                        use_idf=True, 
                                        tokenizer=super().tokenize_and_stem, 
                                        ngram_range=self.ngram_range)

     
        
    def fit_transform(self) -> Dict:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.instance.data["text"])
        terms = self.tfidf_vectorizer.get_feature_names_out()
        return tfidf_matrix,terms


class ClusterAnalyze():

    def __init__(self) -> None:
        self.instance_vector = TextVectorizer()
        self.tfidf_matrix,self.terms = self.instance_vector.fit_transform()
        self.data = self.instance_vector.instance.data
        self.nbr_cluster = self.find_best_clusters()
        self.data["Cluster"] = self.cluster_model().labels_

    def find_best_clusters(self):
        max_clusters = len(self.data)
        best_score = -1
        best_k = 0
        for k in range(2, max_clusters+1):
            kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=1, random_state=42,verbose=0)
            kmeans.fit(self.tfidf_matrix)
            score = silhouette_score(self.tfidf_matrix, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k
    
    def cluster_model(self):
        model = KMeans(n_clusters=self.nbr_cluster, init='k-means++', 
            max_iter=100, n_init=1, verbose=0, random_state=42)
        model.fit_transform(self.tfidf_matrix)
        model.fit(self.tfidf_matrix)
        return model

class SearchEngine(TextPrecessor):
    
    def __init__(self,search:str) -> None:
        self.top_terms_per_cluster = {}
        self.stemming = PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.instance = ClusterAnalyze()
        self.data = self.instance.instance_vector.instance.data
        self.km = KMeans(n_clusters=self.instance.nbr_cluster, max_iter=1000, n_init=1,random_state=42,verbose=0)
        self.km.fit(self.instance.tfidf_matrix)
        self.order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]
        self.search = search 
        self.make_top_terms_per_cluster()
        self.tf_idf_similarity()


    def make_top_terms_per_cluster(self):
        for i in range(self.instance.nbr_cluster):
            self.top_terms_per_cluster[i] = [] 
            for ind in self.order_centroids[i, :100]:
                self.top_terms_per_cluster[i].append(self.instance.terms[ind])
    
    def tf_idf_similarity(self):
        query = self.search
        query_tfidf = self.instance.instance_vector.tfidf_vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_tfidf, self.instance.tfidf_matrix)
        # Classement des articles par ordre décroissant de similarité
        similarity_scores = similarity_scores[0][np.argsort(self.data.index)]
        self.data['similarity'] = list(similarity_scores)
        # Sort the self.dataframe by document ID or title
        self.data = self.data.sort_values('titre')
        # Reset the index to ensure that the order of the documents is always the same
        self.data = self.data.reset_index(drop=True)
        index_max=int(np.argmax(self.data["similarity"]))
        cluster_index_max=self.data["Cluster"][index_max]
        # Affichage des articles similaires
        new = pd.DataFrame(self.data.loc[self.data["Cluster"]==cluster_index_max,["titre"]])
        return list(new["titre"])
    def complex_search(self):
        result = []
        # define a dictionary of synonyms
        synonyms = {}
        for term in re.findall(r'\b\w+\b', self.search):
            synonyms[term] = set()
            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    synonyms[term].add(lemma.name())

        or_keywords = [k.strip() for k in self.search.split("and")]
        and_keywords = []
        for match in re.findall(r'\((.*?)\)|(\b\w+\b)|(\b(or)\b)|(\b(and)\b)', self.search):
            keyword = "".join([x for x in match if x]).strip().lower()
            if keyword:
                if keyword != "or" and "or" in keyword:
                    and_keywords.append([[PorterStemmer().stem(w.strip().lower())]+list(synonyms.get(w.strip().lower(), set())) for w in keyword.split("or")])
                elif keyword != "and" and "and" in keyword:
                    and_keywords.append([[PorterStemmer().stem(w.strip().lower())]+list(synonyms.get(w.strip().lower(), set())) for w in keyword.split("and")])
                else:
                    and_keywords.append([[PorterStemmer().stem(keyword)]+list(synonyms.get(keyword, set()))])

        and_keywords = [x for x in and_keywords if  x != [['or']] and x!=[['and']]]

        for i in range(len(self.data["text"])):
            matches = []
            if len(or_keywords) == 1:
                k = or_keywords[0]
                or_matches = []
                for w in k.split("or"):
                    stemmed_text = [PorterStemmer().stem(word.lower()) for word in self.data["text"][i].split()]
                    found = any([t in stemmed_text for t in and_keywords[0]])
                    or_matches.append(found)
                matches.append(any(or_matches))
            elif len(and_keywords) == 1:
                k = and_keywords[0]
                and_matches = []
                for w in k:
                    found = any([t in str(PorterStemmer().stem(self.data["text"][i].lower())) for t in w])
                    and_matches.append(found)
                matches.append(all(and_matches))
            else:
                for k in and_keywords:
                    or_matches = []
                    for w in k:
                        found = any([t in str(PorterStemmer().stem(self.data["text"][i].lower())) for t in w])
                        or_matches.append(found)
                    matches.append(any(or_matches))
                #matches = [m for m in matches if m]
        
            if all(matches):
                result.append(self.data["titre"][i])
        return result


if __name__ == "__main__":
   instance = SearchEngine("(xgboost or fragmentation) and Nephrolithiasis  and nuclear")
  
   print(instance.complex_search())
    
    
    