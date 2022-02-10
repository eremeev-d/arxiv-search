import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
import json
import string
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool


class Article:
    def __init__(self, article_id, title, abstract):
        self.id = article_id
        self.title = title
        self.abstract = abstract
    
    def format(self, query):
        # возвращает тройку тайтл-текст-ссылка, отформатированную под запрос
        return [
            self.title,
            self.abstract[:500] + ' ...',
            'https://www.arxiv.org/abs/{}'.format(self.id)
        ]


class SearchEngine:
    def __init__(self):
        self.inv_index = {} # inv_index[word] = [doc_id_0, doc_id_1, ..., doc_id_n]
        self.articles_dict = {} # article_dict[id] = article with needed id
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger')
        # TODO: TF-IDF !!!!

    def get_wordnet_pos(self, treebank_tag):
        my_switch = {
            'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.ADV,
        }
        for key, item in my_switch.items():
            if treebank_tag.startswith(key):
                return item
        return wordnet.NOUN

    def lemmatize_text(self, text):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokenized_text = nltk.word_tokenize(text)
        pos_tagged = [(word, self.get_wordnet_pos(tag))
                    for word, tag in pos_tag(tokenized_text)]
        return ' '.join([lemmatizer.lemmatize(word, tag)
                        for word, tag in pos_tagged])

    def tokenize(self, text):
        text = ' '.join(text.split()) # Remove '\n' from text
        text = self.lemmatize_text(text)
        remove_punctuation=str.maketrans('', '', string.punctuation) # Remove punctuation
        text = text.translate(remove_punctuation)
        return text.lower().split()

    def build_index(self):
        # считывает сырые данные и строит индекс
        # Считаем данные, обработаем тексты, сделаем инвертированный индекс

        # Get articles
        articles = []
        with open('Data/articles.json') as data:
            for line in data:
                article = json.loads(line)
                article = Article(article['id'], article['title'], article['abstract'])
                articles.append(article)
                self.articles_dict[article.id] = article

        # Process each article into (id, tokens)    
        articles = map(lambda article: (article.id, set(self.tokenize(article.title)).union(self.tokenize(article.abstract))), articles)
        articles = list(articles)

        # Inverted index
        for article in articles:
            article_id = article[0]
            text = article[1]
            for word in text:
                if not (word in self.inv_index):
                    self.inv_index[word] = set()
                self.inv_index[word].add(article_id)

    def score(self, query, article):
        # возвращает какой-то скор для пары запрос-документ
        # больше -- релевантнее
        # TODO: TF-IDF !!!!
        query_tokens = self.tokenize(query)
        abstract_tokens = self.tokenize(article.abstract)
        score = 0
        for word in query_tokens:
            score += abstract_tokens.count(word)
        return score

    def retrieve(self, query):
        # возвращает начальный список релевантных документов
        words = self.tokenize(query)
        if len(words) == 0:
            return []
        if not (words[0] in self.inv_index):
            return []
        article_idx = self.inv_index[words[0]]
        for word in words[1:]:
            if not (word in self.inv_index):
                return []
            article_idx = article_idx.intersection(self.inv_index[word])
        articles = list(map(lambda article_id: self.articles_dict[article_id], article_idx))
        return articles[:1000]