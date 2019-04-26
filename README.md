# MarketQualityParams
---

The code in this repository uses Trade and Quote data (TAQ) in order to calculate and visualize intraday market quality parameters such as volume, volatility, mid-price and spread. This file summarizes the functions implemented in the calculations of these parameters, as well as the functions for the visualizations generated.

## Prerequisites
---

 * Dask: Parallel coding library along with its dependencies.
 * Scipy: Scientific coding with Python.
 * Numpy: Vector and matrix algebra.
 * Pandas: Data analysis library.
 * `datetime` module: Basic date and time types.

## Data preprocessing
---

In order to calculate the different parameters of interest, our data must be preprocessed and standardized. The preprocessing pipeline is presented in the `DaskPreprocessing.ipynb` jupyter notebook.

 1. The first function in this notebook is the `GetDate()` function, which takes a dataframe of stock data as input. Then, the function outputs a pandas dataframe consisiting of the unique days for which the `stockdata` dataframe contains information. 

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
 
```Python

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
        
        bid = stockdailydata['tipo'] == 'BID'
        ask = stockdailydata['tipo'] == 'ASK'
        BA_df.BID = np.multiply(bid.values, stockdailydata.precio.values)
        BA_df.ASK = np.multiply(ask.values, stockdailydata.precio.values)
        
        
        BA_df['BID'] = BA_df['BID'].replace(to_replace = 0, method = 'ffill').values
        BA_df['ASK'] = BA_df['ASK'].replace(to_replace = 0, method = 'ffill').values
        
        BA_df = BA_df.where(BA_df.BID <= BA_df.ASK, np.nan)
        
        BA_df['Mid_price']     = 0.5*(BA_df['BID'].values + BA_df['ASK'].values)
        BA_df['Quoted_Spread'] = (BA_df['ASK'].values - BA_df['BID'].values)/(BA_df.Mid_price.values)
                    
        BA.append(BA_df)
    
    BA = pd.concat(BA, axis=0)
    BA.index = stockdata.index
    stockdata = pd.concat([stockdata, BA], axis=1)
        
    return stockdata
```

## Stock depth
---

Once we have standardized our data by applying the two functions defined above, we can begin to calculate market quality parameters. In order to determine intraday market depth, we use the `Dask` library to parallelize dataframe operations. The parallelization procedure is summarized in three steps: first, we separate our stock data dataframe into several daily blocks of information. This is done with the `sep_date()` function. Second, the depth for each day is computed by the `DailyDepth()` function. Finally, both of these previous functions are then applied in the `StockDepth()` function.

 1. The `sep_date()` function takes the `stockdata` dataframe as input and outputs a list which entries correspond to the TAQ data of a stock on a given day. 
 
```Python
def sep_date(stockdata):
    days = GetDate(stockdata)
    daily_dfs = []
    for i in days:
        daily_dfs.append(stockdata.loc[stockdata["dia"] == i])
    return daily_dfs
```

 2. The `DailyDepth()` function works out the depth for each dataframe provided as an input as lazy evaluation. It is decorated by the `@delayed` command, which delays the actual computation until the `compute()` method of the Dask library is invoked.
 
```Python
@delayed
def DailyDepth(stockdailydata):
    
    # Creamos columnas para las variables de profundidad
    init_values = np.zeros( np.shape(stockdailydata)[0] )
    #vol = stockdailydata.volumen
    stockdailydata = stockdailydata.assign(**{'BID_depth': init_values, 'ASK_depth': init_values,
                          'Depth': init_values, 'log_depth': init_values})
    
    for j in range(1, np.shape(stockdailydata)[0] ):
        
        #Tipo BID
        if(stockdailydata.tipo[j]=="BID"):
            stockdailydata.ASK_depth[j] = stockdailydata.ASK_depth[j-1]     
            if(stockdailydata.precio[j] == stockdailydata.BID[j]):     
                if(stockdailydata.precio[j] == stockdailydata.BID[j-1]):
                    stockdailydata.BID_depth[j] = stockdailydata.BID_depth[j-1] + stockdailydata.volumen[j]
                elif(stockdailydata.precio[j] != stockdailydata.BID[j-1]):
                    stockdailydata.BID_depth[j] = stockdailydata.volumen[j]
            elif(stockdailydata.precio[j] != stockdailydata.BID[j]):
                stockdailydata.BID_depth[j] = stockdailydata.BID_depth[j-1]   
                    
        #Tipo ASK
        elif(stockdailydata.tipo[j]=="ASK"):
            stockdailydata.BID_depth[j] = stockdailydata.BID_depth[j-1]       
            if(stockdailydata.precio[j] == stockdailydata.ASK[j]):
                if(stockdailydata.precio[j] == stockdailydata.ASK[j-1]):
                    stockdailydata.ASK_depth[j] = stockdailydata.ASK_depth[j-1] + stockdailydata.volumen[j]
                elif(stockdailydata.precio[j] != stockdailydata.ASK[j-1]):
                    stockdailydata.ASK_depth[j] = stockdailydata.volumen[j]
            elif(stockdailydata.precio[j] != stockdailydata.ASK[j]):
                stockdailydata.ASK_depth[j] = stockdailydata.ASK_depth[j-1]
                
        #Tipo TRADE
        elif(stockdailydata.tipo[j]=="TRADE"):
            if(stockdailydata.precio[j] == stockdailydata.ASK[j]):
                stockdailydata.BID_depth[j] = stockdailydata.BID_depth[j-1]
                stockdailydata.ASK_depth[j] = stockdailydata.ASK_depth[j-1] - stockdailydata.volumen[j]
            elif(stockdailydata.precio[j] == stockdailydata.BID[j]):
                stockdailydata.BID_depth[j] = stockdailydata.BID_depth[j-1] - stockdailydata.volumen[j]
                stockdailydata.ASK_depth[j] = stockdailydata.ASK_depth[j-1]
            else:
                stockdailydata.BID_depth[j] = stockdailydata.BID_depth[j-1]
                stockdailydata.ASK_depth[j] = stockdailydata.ASK_depth[j-1]
                
    print("Comienzan a revisarse las condiciones")
    
    # Eliminamos los datos que no tienen sentido
    for k in range(np.shape(stockdailydata)[0]):
        if( stockdailydata.BID_depth[k] < 0):
            stockdailydata.BID_depth[k] = 0
            
        if( stockdailydata.ASK_depth[k] < 0):
            stockdailydata.ASK_depth[k] = 0
            
        # Se calcula la profundidad
        stockdailydata.Depth[k] = stockdailydata.BID_depth[k] + stockdailydata.ASK_depth[k]
        
        # Se calcula la log-profundidad
        if(stockdailydata.ASK_depth[k] != 0 and stockdailydata.BID_depth[k] != 0):
            stockdailydata.log_depth[k] = np.log(stockdailydata.BID_depth[k] * stockdailydata.ASK_depth[k])
            
    # Se quitan los NaN de los datos de profundidad
    for l in range(0, len(stockdailydata.tipo)):   
        if (np.isnan(stockdailydata.Quoted_Spread[l]) == True): 
            stockdailydata.BID_depth[l] = 0
            stockdailydata.ASK_depth[l] = 0
            stockdailydata.Depth[l]     = 0
            stockdailydata.log_depth[l] = 0
            
    return stockdailydata
```

 3. The previous two methods are utilized in the `StockDepth()` function. This takes the `stockdata` pandas dataframe as input, subsequently applying the parallel pipeline described previously in order to output a dask dataframe that contains the stock's depth.

```Python
def StockDepth(stockdata):
    daily_df = sep_date(stockdata)
    delayed_dfs = []
    
    for df in daily_df:
        delayed_dfs.append( DailyDepth(df) )
        
    result_df = dd.from_delayed(delayed_dfs)
    return result_df.compute()
```

## Buy-sell
---

```Python
def InitiatingParty(stockdata):
    '''
    Parameters:
    ------
    stockdata:
    DataFrame - Data of the stock
    
    Return:
    ------
    x:
    DataFrame - DataFrame of TRADE quotes with the party that initiated the trade
    '''
    
    x = stockdata[stockdata.tipo == 'TRADE']
    
    # +1: transaccion iniciada por comprador
    buyer  = x.precio.values > x.Mid_price.values
    
    # -1: transaccion iniciada por vendedor
    seller = x.precio.values < x.Mid_price.values
    
    x['iniciado'] = buyer.astype(int) - seller.astype(int)
    x['iniciado'] = x['iniciado'].replace(to_replace = 0, method = 'ffill').values
    
    return x
```

## Price impact
---

```Python
def ImpactParameters(stockdata):
    days = GetDate(stockdata)
    res = []
    
    for i in days:
        stockdailydata = stockdata[stockdata.dia == str(i)]
        
        stockdailydata['delta_p']    = stockdailydata['precio'].diff()
        stockdailydata['order_flow'] = stockdailydata.volumen.values * stockdailydata.iniciado.values
        
        res.append(stockdailydata)
        
    res_df = pd.concat(res, axis=0)
    return res_df
```

```Python
def KyleImpactRegression(stockdata):
    
    days = GetDate(stockdata)#.drop_duplicates(keep='first').dia
    res = []
    
    for i in days:
        
        stockdailydata = stockdata[stockdata.dia == str(i)]
        
        x1 = stockdailydata.delta_p.values
        x1 = x1.reshape(-1, 1)
        
        x2 = stockdailydata.order_flow.values
        x2 = sm.add_constant(x2.reshape(-1, 1))
        
        result = sm.OLS(x1, x2, missing='drop').fit()
        
        coef = result.params[1]
        pvalue = result.pvalues[1]
        trades = len(stockdailydata)
        
        temp = [i, coef, pvalue, trades]
        res.append(temp)
        
    #res = pd.DataFrame(res, columns=['day', 'reg_coefficient', 'p_value', 'trades'])
    res = pd.DataFrame(res, columns=['dia', 'coef_regresion', 'p_value', 'trades'])
    res = res.set_index('dia')
    
    return res
```

Calculating and visualizing market quality parameters
