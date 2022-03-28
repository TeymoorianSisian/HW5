import spacy
import en_core_web_lg
import pickle
import pandas as pd
from spacy.lang.en import English
from collections import Counter
from string import punctuation
from wordcloud import WordCloud
from newsapi import NewsApiClient
import matplotlib.pyplot as plt

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient(api_key='fa902e51b3294bd380a97c53aebb3cf6')
articles =[]

for pagina in range(5):
    temp = newsapi.get_everything(q='coronavirus',language='en',from_param='2022-03-01',to='2022-03-20',
                                  sort_by='relevancy',page=pagina+1)
    articles.append(temp)

filename = 'articlesCOVID.pckl'
pickle.dump(articles,open(filename,'wb'))
filename ='articlesCOVID.pckl'
loaded_model = pickle.load(open(filename,'rb'))
filepath = "articlesCOVID.pckl"
pickle.dump(loaded_model, open(filepath,'wb'))

dados = []

for i, article in enumerate(articles):
    for x in article['articles']:
        title = x['title']
        description = x['description']
        content = x['content']
        dados.append({'title':title,'desc':description,'content':content})

df = pd.DataFrame(dados)
df = df.dropna()
df.head()

results = []

def get_keywords_eng(text):
    result =[]
    pos_tag =['NOUN','VERB','PROPN']
    doc = nlp_eng(text)
    for token in doc:
        if(token.text in nlp_eng.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    print(result) #check to make sure it works
    return result

for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])

df['keywords'] = results
pickle.dump(df,open(filename,'wb'))
df.to_csv(r'covid.csv',index=0)
df.head()
print(df[['content','keywords']])
#wordcloud
text = str(results)
wordcloud = WordCloud(max_font_size=50,max_words=100,background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
