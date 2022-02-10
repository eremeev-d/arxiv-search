import random
import json
import string


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
        # TODO: TF-IDF !!!!

    def tokenize(self, text):
        text = ' '.join(text.split()) # Remove '\n' from text
        remove_punctuation=str.maketrans('', '', string.punctuation) # Remove punctuation
        text = text.translate(remove_punctuation)
        return text.lower().split()

    def build_index(self):
        # считывает сырые данные и строит индекс
        with open('Data/articles.json') as data:
            for line in data:
                article = json.loads(line)
                article = Article(article['id'], article['title'], article['abstract'])
                self.articles_dict[article.id] = article
                words = set(self.tokenize(article.title))
                words = words.union(self.tokenize(article.abstract))
                for word in words:
                    if not (word in self.inv_index):
                        self.inv_index[word] = set()
                    self.inv_index[word].add(article.id)

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
        # TODO: убирать знаки препинания!
        words = self.tokenize(query)
        if len(words) == 0:
            return []
        article_idx = self.inv_index[words[0]]
        for word in words[1:]:
            article_idx = article_idx.intersection(self.inv_index[word])
        articles = list(map(lambda article_id: self.articles_dict[article_id], article_idx))
        return articles[:10000]