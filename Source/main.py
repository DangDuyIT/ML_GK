import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from underthesea import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import codecs
from sklearn.metrics import accuracy_score
from underthesea.transformer.count import CountVectorizer

topic1 = "https://vnexpress.net/kinh-doanh"
topic2 = "https://vnexpress.net/giai-tri"
topic3 = "https://vnexpress.net/giao-duc"
topic4 = "https://vnexpress.net/the-thao"
topic5 = "https://vnexpress.net/phap-luat"

def Topic1(topic):
    data = []
    response = requests.get(topic)
    soup = BeautifulSoup(response.content, "html.parser")
    titles = soup.findAll('h3', {"class": ["title-news", "title_news"]},limit=50)
    links = [link.find('a').attrs["href"] for link in titles]
    dataDetail = []
    for link in links:
        news = requests.get(link)
        soup = BeautifulSoup(news.content, "html.parser")
        if hasattr(soup.find("h1", class_="title-detail"), 'text'):
            CrawlTitles = soup.find("h1", class_="title-detail").text
        else:
            CrawlTitles = ""
        if hasattr(soup.find("p", class_="description"), 'text'):
            CrawlDesc = soup.find("p", class_="description").text
        else:
            CrawlDesc = ""

        CrawlBody = soup.find("article", class_="fck_detail")
        CrawlContent = ""
        try:
            CrawlContent = CrawlBody.findChildren("p", recursive=False)[0].text + CrawlBody.findChildren("p", recursive=False)[
                    1].text
        except:
            CrawlContent = ""

        dataDetail.append({CrawlTitles ,CrawlDesc, CrawlContent,})

    for i in dataDetail:
        data.append(i.__str__())
    pd.DataFrame(data).to_csv('Topic1.csv',encoding='utf-8-sig', index=False,header=True)
def Topic2(topic):
    data = []
    response = requests.get(topic)
    soup = BeautifulSoup(response.content, "html.parser")
    titles = soup.findAll('h3', {"class": ["title-news", "title_news"]},limit=50)
    links = [link.find('a').attrs["href"] for link in titles]
    dataDetail = []
    for link in links:
        news = requests.get(link)
        soup = BeautifulSoup(news.content, "html.parser")
        if hasattr(soup.find("h1", class_="title-detail"), 'text'):
            CrawlTitles = soup.find("h1", class_="title-detail").text
        else:
            CrawlTitles = ""
        if hasattr(soup.find("p", class_="description"), 'text'):
            CrawlDesc = soup.find("p", class_="description").text
        else:
            CrawlDesc = ""

        CrawlBody = soup.find("article", class_="fck_detail")
        CrawlContent = ""
        try:
            CrawlContent = CrawlBody.findChildren("p", recursive=False)[0].text + CrawlBody.findChildren("p", recursive=False)[
                    1].text
        except:
            CrawlContent = ""

        dataDetail.append({CrawlTitles ,CrawlDesc, CrawlContent,})

    for i in dataDetail:
        data.append(i.__str__())
    pd.DataFrame(data).to_csv('Topic2.csv',encoding='utf-8-sig', index=False,header=True)
def Topic3(topic):
    data = []
    response = requests.get(topic)
    soup = BeautifulSoup(response.content, "html.parser")
    titles = soup.findAll('h3', {"class": ["title-news", "title_news"]},limit=50)
    links = [link.find('a').attrs["href"] for link in titles]
    dataDetail = []
    for link in links:
        news = requests.get(link)
        soup = BeautifulSoup(news.content, "html.parser")
        if hasattr(soup.find("h1", class_="title-detail"), 'text'):
            CrawlTitles = soup.find("h1", class_="title-detail").text
        else:
            CrawlTitles = ""
        if hasattr(soup.find("p", class_="description"), 'text'):
            CrawlDesc = soup.find("p", class_="description").text
        else:
            CrawlDesc = ""

        CrawlBody = soup.find("article", class_="fck_detail")
        CrawlContent = ""
        try:
            CrawlContent = CrawlBody.findChildren("p", recursive=False)[0].text + CrawlBody.findChildren("p", recursive=False)[
                    1].text
        except:
            CrawlContent = ""

        dataDetail.append({CrawlTitles ,CrawlDesc, CrawlContent,})

    for i in dataDetail:
        data.append(i.__str__())
    pd.DataFrame(data).to_csv('Topic3.csv',encoding='utf-8-sig', index=False,header=True)
def Topic4(topic):
    data = []
    response = requests.get(topic)
    soup = BeautifulSoup(response.content, "html.parser")
    titles = soup.findAll('h3', {"class": ["title-news", "title_news"]},limit=50)
    links = [link.find('a').attrs["href"] for link in titles]
    dataDetail = []
    for link in links:
        news = requests.get(link)
        soup = BeautifulSoup(news.content, "html.parser")
        if hasattr(soup.find("h1", class_="title-detail"), 'text'):
            CrawlTitles = soup.find("h1", class_="title-detail").text
        else:
            CrawlTitles = ""
        if hasattr(soup.find("p", class_="description"), 'text'):
            CrawlDesc = soup.find("p", class_="description").text
        else:
            CrawlDesc = ""

        CrawlBody = soup.find("article", class_="fck_detail")
        CrawlContent = ""
        try:
            CrawlContent = CrawlBody.findChildren("p", recursive=False)[0].text + CrawlBody.findChildren("p", recursive=False)[
                    1].text
        except:
            CrawlContent = ""

        dataDetail.append({CrawlTitles ,CrawlDesc, CrawlContent,})

    for i in dataDetail:
        data.append(i.__str__())
    pd.DataFrame(data).to_csv('Topic4.csv',encoding='utf-8-sig', index=False,header=True)
def Topic5(topic):
    data = []
    response = requests.get(topic)
    soup = BeautifulSoup(response.content, "html.parser")
    titles = soup.findAll('h3', {"class": ["title-news", "title_news"]},limit=50)
    links = [link.find('a').attrs["href"] for link in titles]
    dataDetail = []
    for link in links:
        news = requests.get(link)
        soup = BeautifulSoup(news.content, "html.parser")
        if hasattr(soup.find("h1", class_="title-detail"), 'text'):
            CrawlTitles = soup.find("h1", class_="title-detail").text
        else:
            CrawlTitles = ""
        if hasattr(soup.find("p", class_="description"), 'text'):
            CrawlDesc = soup.find("p", class_="description").text
        else:
            CrawlDesc = ""

        CrawlBody = soup.find("article", class_="fck_detail")
        CrawlContent = ""
        try:
            CrawlContent = CrawlBody.findChildren("p", recursive=False)[0].text + CrawlBody.findChildren("p", recursive=False)[
                    1].text
        except:
            CrawlContent = ""

        dataDetail.append({CrawlTitles ,CrawlDesc, CrawlContent,})

    for i in dataDetail:
        data.append(i.__str__())
    pd.DataFrame(data).to_csv('Topic5.csv',encoding='utf-8-sig', index=False,header=True)


df1 = pd.read_csv("Topic1.csv")
df1.insert(0,"Label","Kinh Doanh")
df2 = pd.read_csv("Topic2.csv")
df2.insert(0,"Label","Giải Trí")
df3 = pd.read_csv("Topic3.csv")
df3.insert(0,"Label","Thể Thao")
df4 = pd.read_csv("Topic4.csv")
df4.insert(0,"Label","Pháp Luật")
df5 = pd.read_csv("Topic5.csv")
df5.insert(0,"Label","Giáo Dục")

frames = [df1,df2,df3,df4,df5]
pd.DataFrame(pd.concat(frames).to_csv('AllData.csv',encoding='utf-8-sig',index = False,header=True))
df = pd.read_csv("AllData.csv")

#Remove missing
df = df.dropna(axis=0, how='any')
#Token
allData = df.to_string()
Token = word_tokenize(allData)
#lowerCase
for i in range(len(Token)):
    Token[i] = Token[i].lower()
#remove stop word
f = codecs.open('vietnamese-stopwords.txt', encoding='utf-8')
stopWord = f.readlines.__str__()
removeStopWord = [i for i in Token if not i in stopWord]

FinalWord = removeStopWord


wordSet = set(FinalWord)
wordDict = dict.fromkeys(wordSet,0)
for word in FinalWord:
    wordDict[word] +=1
#Compute TF
TF = {}
wordsCount = len(FinalWord)
for word, count in wordDict.items():
    TF[word] = count/float(wordsCount)

#Compute IDF
IDF = {}
wordList = [wordDict]
IDF_Dict = dict.fromkeys(wordList[0].keys(), 0)
for doc in wordList:
    for word, val in doc.items():
        if val > 0:
            IDF_Dict[word] =+1
for word, val in IDF_Dict.items():
    IDF_Dict[word] = np.math.log10(len(wordList)/float(val))

#Compute TF-IDF
TF_IDF = {}
for words, val in TF.items():
    TF_IDF[words] = val* IDF_Dict[words]
TF_IDF = pd.DataFrame([TF_IDF])


#Divide train/test

text = []
label = []

text = df.iloc[:,1]
label = df.iloc[:,0]



X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)
# encode label
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
#Training SVM

train= Pipeline([('Vector', CountVectorizer()),
                     ('TD-IDF', TfidfTransformer()),
                     ('SVC', SVC(gamma='scale'))])
train = train.fit(X_train, y_train)

y_pred = train.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))
# if __name__ == "__main__":
#     Topic1(topic1)
#     Topic2(topic2)
#     Topic3(topic3)
#     Topic4(topic4)
#     Topic5(topic5)


