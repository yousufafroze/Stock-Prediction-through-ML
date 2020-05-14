'''
Credits - Used Python Programming for Finance tutorials from sentdex
'''

from collections import Counter
import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from pandas.util.testing import assert_frame_equal
import pickle
import requests
import matplotlib.pyplot as plt
from matplotlib import style


from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


'''
Will consider other stocks, since the stocks of a group of
companies have a high correlation. They don't all move together, hence
we can maybe take advantage of the lag that some stocks have

We will change everything to percentage, to normalize the data

'''
## Returns a list of all available S&P 500 tickers from wikipedia
def save_sp500_tickers():
    resp  = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class":"wikitable sortable"})
    tickers = []
    for row in table.findAll("tr")[1:]: ## To avoid taking the header of the table
        ticker = row.findAll("td")[0].text.rstrip()
        if ticker not in ['BRK.B', 'BF.B', 'CARR', 'CTVA', 'DOW', 'FOXA', 'FOX', 'HWM', 'IR', 'NLOK', 'OTIS', 'VIAC']:
            tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


## Makes csv files to store information for available S&P 500 tickers 
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists("stock_dfs"):
        os.makedirs("stock_dfs")

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2016, 12, 31)
    for ticker in tickers:
        if not os.path.exists("stock_dfs/{}.csv".format(ticker)):
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv("stock_dfs/{}.csv".format(ticker))
        else:
            print("Already have {}".format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()


    for count, ticker in enumerate(tickers):
        df = pd.read_csv("stock_dfs/{}.csv".format(ticker))
        df.set_index("Date", inplace=True)

        df.rename(columns = {"Adj Close": ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how="outer")

        if count%10==0:
            print(count)
        
    main_df.to_csv('sp500_joined_closes.csv')


## Visualize the correlation between the SP 500 stocks in a heatmap
def visualize_data():
    df = pd.read_csv("sp500_joined_closes.csv")

    ## Calculate correlation between stocks for proper portfolio diversification
    ## Also, for capital gain opportunities
    df_corr = df.corr() 

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn) # Red|Yellow|Green -> Negative|Neutral|Positive
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
 
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


def process_data_for_labels(ticker):
    hm_days = 7 ## Since 7 seems to be a good amound of days to see correlation
    df = pd.read_csv("sp500_joined_closes.csv", index_col=0)

    tickers = df.columns.values
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]

    ## High requirement gives off a higher accuracy, since it's less prone to unreasoned deviation
    requirement = 0.05 ## Deviation tolerance

    for col in cols:
        if col > requirement:
            return 1
        elif col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)


    df["{}_target".format(ticker)] = list(map(buy_sell_hold, 
                                              df['{}_1d'.format(ticker)], 
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print("Data spread:", Counter(str_vals))
 
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # Not same as df, it doesn't include future day price changes    
    ## To percent change, normalizes the data for fair comparison. 
    df_vals = df[[ticker for ticker in tickers]].pct_change() 
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)


    X = df_vals.values
    y = df['{}_target'.format(ticker)].values ## With values: -1,0,+1 

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    print("Accuracy", confidence)
    predictions = clf.predict(X_test)
    print("Predicted spread:", Counter(predictions))
    print("Actual spread:", Counter(y_test))

    return confidence



do_ml("SHW")

 
