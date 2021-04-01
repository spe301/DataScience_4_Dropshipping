from pytrends.request import TrendReq
import tweepy
from statsmodels.tsa import ar_model, stattools, arima_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from textblob import TextBlob
import pandas as pd
from potosnail import Stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from scipy.signal import find_peaks
import datetime
from flask import Flask, request, jsonify, render_template

def GetReport(keywords, span='today 5-y', geo='', quiet=True):
    '''observe a search term's popularity in the past 5 years'''
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(keywords, cat=0, timeframe=span, geo=geo, gprop='')
    ts = pytrends.interest_over_time().drop(['isPartial'], axis='columns')
    if quiet == False:
        print(ts.plot())
    return ts

def AnalyzeTwitter(keyword):
    '''find the average sentimental value and subjectivity of a given search term'''
    c1 = 'aHXduTrDkva3ItY52tUtYVPvA'
    c2 = 'Qs6d4oNT3zXxDqOhita7IG07CfAJGceoqIs1sGuA4OURlbLP6d'
    a1 = '1181578611171762177-sGQaj7E9fpWi2aEB3MfWL4nTRovXYk'
    a2 = 'wa77yBJZJSOKOAzdaJYDruc9U1HrGhzyDhWgKvSQpm2hv'
    auth = tweepy.OAuthHandler(c1, c2)
    auth.set_access_token(a1, a2)
    api = tweepy.API(auth)
    topic = api.search(keyword)
    sent = 0
    sub = 0
    sents = []
    for i in range(len(topic)):
        tweet = topic[i]._json['text'].replace('@', '')
        blob = TextBlob(tweet)
        sents.append(blob.sentiment[0])
        sent += blob.sentiment[0]/len(topic)
        sub += blob.sentiment[1]/len(topic)
    return sent, sub, sents

def Collect(keyword, quiet=True):
    row = {}
    tsdf = BuildTS(keyword)
    row['term'] = keyword
    current_popularity = list(tsdf[keyword][:260])[-1]
    row['current_popularity'] =  current_popularity
    row['change_3mo'] = '{}%'.format(round(((tsdf[keyword][271] - current_popularity) / current_popularity) * 100, 1))
    row['change_6mo'] = '{}%'.format(round(((tsdf[keyword][283] - current_popularity) / current_popularity) * 100, 1))
    row['change_9mo'] = '{}%'.format(round(((tsdf[keyword][295] - current_popularity) / current_popularity) * 100, 1))
    row['change_12mo'] = '{}%'.format(round(((tsdf[keyword][307] - current_popularity) / current_popularity) * 100, 1))
    row['change_24mo'] = '{}%'.format(round(((tsdf[keyword][355] - current_popularity) / current_popularity) * 100, 1))
    try:
        row['popularity_2y'] = round((((tsdf[keyword][355] - current_popularity) / current_popularity) + 1) * current_popularity)
    except:
        row['popularity_2y'] = round(tsdf[keyword][355])
    sentiment, subjectivity, sentiments = AnalyzeTwitter(keyword)
    row['sentiment'] = round(sentiment, 2)
    row['subjectivity'] = round(subjectivity, 2)
    row['sentiments_std'] = round(np.std(sentiments), 2)
    if quiet == True:
        return row
    else:
        return tsdf, row

def CollectLoop(terms_list):
    '''tells us how popularity for a given list of search terms are expected to change'''
    df = pd.DataFrame(Collect(terms_list[0]), index=[0])
    for term in terms_list[1:]:
        temp = pd.DataFrame(Collect(term), index=[0])
        df = pd.concat([df, temp])
    return df.reset_index().drop(['index'], axis='columns')

def PlotOne(keyword):
    '''the output a user gets when looking at one term'''
    ts, results = Collect(keyword, quiet=False)
    subj = results['subjectivity']
    obj = 1 - subj
    X = ['%subjective', '%objective']
    y = [subj, obj]
    X2 = ['sentiment']
    y2 = results['sentiment']
    if results['popularity_2y'] > results['current_popularity']:
        future = 'increase'
    else:
        future = 'decrease'
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = results['sentiment'],
    mode = "gauge+number",
    title = {'text': "Sentiment of '{}' based on tweets".format(keyword)},
    gauge = {'axis': {'range': [-1, 1]},
             'steps' : [
                 {'range': [-1, 0], 'color': "red"},
                 {'range': [0, 1], 'color': "lightgreen"}]}))
    fig.show()
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = results['subjectivity'],
    mode = "gauge+number",
    title = {'text': "Subjectivity of '{}' based on tweets".format(keyword)},
    gauge = {'axis': {'range': [0, 1]},
             'steps' : [
                 {'range': [0, 0.5], 'color': "yellow"},
                 {'range': [0.5, 1], 'color': "blue"}]}))
    fig.show()
    fig = px.line(ts, x='index', y=keyword, range_y=[0, 100])
    fig.show()
    
def PlotMany(keywords):
    df = CollectLoop(keywords)
    fig = px.bar(df, x='term', y='current_popularity', color='sentiment', range_y=[0, 100])
    fig.show()
    for i in range(len(keywords)):
        ser = Collect(keywords[i], quiet=False)[0]
        fig = px.line(ser, x='index', y=keywords[i], range_y=[0, 100])
        fig.show()
        
def GetPeaks(ser):
    kw = list(ser.columns)[0]
    varience = 0
    for i in range(len(ser)):
        varience += abs(np.mean(ser)[0] - ser.iloc[i][0])
    delta = abs(np.mean(ser.iloc[235:])[0] - np.mean(ser.iloc[:27])[0])
    si = varience/delta
    x = np.array(list(GetReport([kw])[kw]))
    peaks, _ = find_peaks(x, prominence=10, distance=52)
    return peaks, si
    
def CheckSeasonality(ser, quiet=True):
    peaks, si = GetPeaks(ser)
    n_peaks = len(peaks)
    if quiet == False:
        print(peaks, si)
    if si > 250:
        if n_peaks < 3:
            return False
        else:
            return True
    else:
        if n_peaks > 4:
            return True
        else:
            if CovidCheck(peaks) == False:
                return False
            else:
                p1 = len(GetPeaks(ser.loc[:np.datetime64('2020-03-08')])[0])
                p2 = len(GetPeaks(ser.loc[np.datetime64('2020-07-26'):])[0])
                return p1+p2 > 4
    
def BuildTS(keyword):
    ser = GetReport([keyword])
    s = CheckSeasonality(ser)
    if s == True:
        my_order = (2,1,2) #probably wrong, also needs to be programatic
        my_seasonal_order = (2, 1, 2, 52) #probably wrong, also needs to be programatic
        model = SARIMAX(ser, order=my_order, seasonal_order=my_seasonal_order).fit()
        pred = model.predict(start=len(ser), end=356)
        ser_ = pd.DataFrame(ser)
        pred_ = pd.DataFrame(pred)
        pred_.columns = [keyword]
        ser_.columns = [keyword]
        return pd.concat([ser_, pred_]).reset_index()
    if s == False:
        model = ar_model.AutoReg(ser, lags=4).fit()
        pred = model.predict(start=len(ser), end=356)
        ser_ = pd.DataFrame(ser)
        pred_ = pd.DataFrame(pred)
        pred_.columns = [keyword]
        ser_.columns = [keyword]
        return pd.concat([ser_, pred_]).reset_index()
    
def PredictSearches(to_predict):
    if type(to_predict) == str:
        return PlotOne(to_predict)
    if type(to_predict) == list:
        if len(to_predict) == 1:
            return PlotOne(to_predict[0])
        else:
            return PlotMany(to_predict)
        
def CovidCheck(peaks):
    today = datetime.date.today().strftime("%Y-%m-%d")
    delta = np.datetime64(today) - np.datetime64('2021-03-28')
    delta = int(delta.astype(int)/7)
    peaks = np.array(peaks)
    spike = np.array(list(range(205-delta, 226-delta)))
    affected = np.intersect1d(spike, peaks)
    regular = 0
    try:
        for peak in affected:
            if affected-52 in list(peaks):
                regular += 1
    except:
        return False
    return len(affected)!=0 and regular==0

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['Text']
    return render_template('index.html', PredictSearches(user_input))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)