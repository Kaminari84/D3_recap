import requests
from pprint import pprint
import json

#https://www.worldtradingdata.com/home

api_token = "bqJzldsXBiJSKwEBslg1DWDiRnFCgiun4VA5SiRnYZN0BwRWwCkHYIYxxkeX"
stock_ticket = "AAPL"

'''r = { 'data':{ 
        '2019-07-24': {'close': '208.67', 
                'high': '209.15', 
                'low': '207.17', 
                'open': '207.67', 
                'volume': '14991567'}, 
        '2019-07-25': {'close': '207.02', 
                'high': '209.24', 
                'low': '206.73', 
                'open': '208.89', 
                'volume': '13909562'}, 
        '2019-07-26': {'close': '207.74', 
                'high': '209.73', 
                'low': '207.14', 
                'open': '207.48',
                'volume': '17618874'} 
        }, 
    'name': 'AAPL'}'''

'''r = requests.get('https://api.worldtradingdata.com/api/v1/history?symbol='+stock_ticket+'&sort=newest&api_token='+api_token)
f = open('AAPL.json', 'w')
f.write(json.dumps(r.json()))
f.close()
pprint(r.json())
'''


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
import matplotlib.dates as mdates

#Load JSON
f = open('AAPL.json', 'r')
r = json.load(f)
f.close()

n = 0
values = []
features = []
dates = []
prices = []

for key in sorted(r['history'].keys(), reverse=True):
    val = r['history'][key]
    #print("Key:",key," ,Val:",val)
    values.append( { 'datetime': key, 'close': val['close'] } )
    
    year = int(key.split('-')[0])
    month = int(key.split('-')[1])
    day = int(key.split('-')[2])

    #prev_price = 0
    #if len(prices)>0:
    #    prev_price = prices[0] #previous day price

    features.insert(0, [year, month, day])
    dates.insert(0, dt.datetime(year, month, day))
    prices.insert(0, float(val['close']))
    n += 1
    if n>120:
        break

print(features)
print(dates)
print(prices)

def predict_prices(dates, features, prices, x):
    #dates = np.reshape(dates, (len(dates), 1))
    features = np.reshape(features, (len(features), 3))
    print(features[:,1:3])
    

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree = 2, gamma='scale')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')

    svr_lin.fit(features[:,0:3], prices)
    svr_poly.fit(features[:,1:3], prices)
    svr_rbf.fit(features[:,0:3], prices)

    #dates = [dt.datetime.fromtimestamp(ts) for ts in timestamps]

    #years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    days = mdates.DayLocator()
    yearsFmt = mdates.DateFormatter('%Y-%m')
    

    fig, ax = plt.subplots()
    plt.plot(dates, prices, 'go--',  c='black', label='Data', linewidth=0.5, markersize=1)#scatter, s=1.5)
    plt.plot(dates, svr_lin.predict(features[:,0:3]), color='green', label='Linear model', linewidth=1.0)
    plt.plot(dates, svr_poly.predict(features[:,1:3]), color='blue', label='Polynomial model', linewidth=0.5)
    plt.plot(dates, svr_rbf.predict(features[:,0:3]), color='red', label='RBF model')

    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(days)

    # round to nearest years...
    #datemin = np.datetime64(dates[0], 'm')
    #datemax = np.datetime64(dates[-1], 'm') + np.timedelta64(1, 'm')
    #ax.set_xlim(datemin, datemax)

    # format the coords message box
    def price(x):
        return '$%1.2f' % x
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = price
    ax.grid(True)


    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()

    fig.autofmt_xdate()
    plt.show()


predict_prices(dates, features, prices, 26)