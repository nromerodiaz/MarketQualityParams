# MarketQualityParams
---

The code in this repository uses Trade and Quote data (TAQ) in order to calculate and visualize intraday market quality parameters such as volume, volatility, mid-price and spread. This file summarizes the functions implemented in the calculations of these parameters, as well as the functions for the visualizations generated.

## Prerequisites
---

 * Dask: Parallel coding library along with its dependencies.
 * Scipy: Scientific coding with Python.
 * Numpy: Vector and Matrix algebra.
 * Pandas: Data analysis library.
 * `datetime` module: Basic date and time types.

## Data preprocessing
---

In order to calculate the different parameters of interest, our data must be preprocessed and stardadized. The preprocessing sequence is presented in the `DaskPreprocessing.ipynb` jupyter notebook.

 1. The first function is the `GetDate()` function. This functions takes a dataframe of stock data as input. Then, the function outputs a pandas dataframe consisiting of the unique days for which the `stockdata` dataframe contains information. 

```python

def GetDate(stockdata):
    '''
    Parameters:
    ------
    stockdata:
    DataFrame - Data of various stocks
    
    Return:
    ------
    days: DataFrame - DataFrame of all days for which there is data
    '''
    
    days = pd.DatetimeIndex(stockdata.index).normalize()
    days = pd.DataFrame(days)
    days.index = stockdata.index
    days.columns = ['dia']
    
    return days.drop_duplicates(keep='first').dia
```

 This function is defined in order to be used within other functions presented in this section.

 2. The `StockPreprocessing()` function simultaneously initializes several columns and market quality parameters for a given stock ticker (`stockticker`) contained within `stockdata`. These columns and parameters are: `name`, `date_time`, `type`, `price`, `volume`, `BID`, `ASK`, `Mid_Price` and `Quoted_Spread` columns. Market quality parameters are computed daily, and are subsequently concatenated into the original `stockdata` dataframe.
 
```python

# Funcion que inicializa las columnas: 'nombre', 'date_time', 'tipo', 'precio', 'volumen',
#                                      'BID', 'ASK', 'Mid_Price', 'Quoted_Spread'

def StockPreprocessing(stockdata, stock_ticker):
    '''
    Parameters:
    ------
    stockdata:
    DataFrame - Data of various stocks
    
    stock_ticker:
    String - Ticker of the stock we are interested in
    
    
    Return:
    ------
    stockdata:
    DataFrame - Data of stocks with the folloeing initialized columns: 
    nombre', 'date_time', 'tipo', 'precio', 'volumen', 'BID', 'ASK', 'Mid_Price', 'Quoted_Spread'
    '''
    
    stockname = stock_ticker + " CB Equity"
    
    #Se cambian los nombres de las columnas y se elimina lo demas
    stockdata = stockdata[['name', 'times', 'type', 'value', 'size']]
    stockdata.columns=['nombre','date_time','tipo','precio','volumen']    
    
    #Se seleccionan los datos segun la accion y el horario que nos interesan
    stockdata = stockdata.loc[(stockdata["nombre"] == stockname)]
    stockdata.index = stockdata.date_time
    stockdata = stockdata.between_time('9:30','15:55')
    stockdata['dia'] = pd.DatetimeIndex(stockdata.date_time).normalize() 
    
    days = GetDate(stockdata)#.drop_duplicates(keep='first').dia
    
    BA = []
    
    for i in days:
        stockdailydata = stockdata[stockdata.dia == str(i)]
        
        init_values = stockdailydata.precio.values
        d = {'BID': init_values, 'ASK': init_values}
        BA_df = pd.DataFrame(data=d)
        #BA_df.index = stockdata.index
        
        bid = stockdailydata['tipo'] == 'BID'
        ask = stockdailydata['tipo'] == 'ASK'
        BA_df.BID = np.multiply(bid.values, stockdailydata.precio.values)
        BA_df.ASK = np.multiply(ask.values, stockdailydata.precio.values)
        
        #BA_df['BID'].replace(to_replace = 0, method = 'ffill')
        #BA_df['ASK'].replace(to_replace = 0, method = 'ffill')
        BA_df['BID'] = BA_df['BID'].replace(to_replace = 0, method = 'ffill').values
        BA_df['ASK'] = BA_df['ASK'].replace(to_replace = 0, method = 'ffill').values
        
        BA_df = BA_df.where(BA_df.BID <= BA_df.ASK, np.nan) #np.nan
        
        BA_df['Mid_price']     = 0.5*(BA_df['BID'].values + BA_df['ASK'].values)
        BA_df['Quoted_Spread'] = (BA_df['ASK'].values - BA_df['BID'].values)/(BA_df.Mid_price.values)
            
        #BA_df.index = stockdailydata.index
        
        BA.append(BA_df)
    
    BA = pd.concat(BA, axis=0)
    BA.index = stockdata.index
    stockdata = pd.concat([stockdata, BA], axis=1)
        
    return stockdata
    
```

Calculating and visualizing market quality parameters
