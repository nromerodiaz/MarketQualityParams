# MarketQualityParams

*Calculating and visualizing market quality parameters*
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

## Visualization
---

Now, we present a summary of the `Visualization` notebook.

Calculating and visualizing market quality parameters

```Python
# Funcion que calcula los dias habiles en Colombia

def business_days():
    # 2017
    weekmask = "Mon Tue Wed Thu Fri"
    holidays = [datetime.datetime(2017, 1, 9), datetime.datetime(2017, 3, 20), datetime.datetime(2017, 4, 13), 
                datetime.datetime(2017, 4, 14), datetime.datetime(2017, 5, 1), datetime.datetime(2017, 5, 29), 
                datetime.datetime(2017, 6, 19), datetime.datetime(2017, 6, 26), datetime.datetime(2017, 7, 3), 
                datetime.datetime(2017, 7, 20), datetime.datetime(2017, 8, 7), datetime.datetime(2017, 8, 21), 
                datetime.datetime(2017, 10, 16), datetime.datetime(2017, 11, 6), datetime.datetime(2017, 11, 13), 
                datetime.datetime(2017, 12, 8), datetime.datetime(2017, 12, 25)]

    BdaysCol2017 = pd.bdate_range(start = pd.datetime(2017, 1, 1), end = pd.datetime(2017, 12, 31), 
                                  freq = 'C',
                                  weekmask = weekmask, 
                                  holidays = holidays)
    
    return BdaysCol2017
```

```Python
# Funcion que formatea los datos de un parametro para una accion

# Esta funcion retorna una matriz indexada por la hora en las filas y por
# dias en las columnas con los datos del parametro "param_name" en el dia 
# y hora correspondientes

def get_stock_param(stockdata, stockticker, param_name):
    
    stockdata = stockdata[stockdata.nombre == stockticker + " CB Equity"]
    
    idx = pd.to_datetime(stockdata.date_time)
    stockdata.index = idx
    stockdata.replace(0, np.nan, inplace=True)
    
    bus_days = business_days()
    
    # Log-depth
    if(param_name == "log_depth"):        
        
        depth15min = stockdata.log_depth.resample("15T").mean()
        depth15min = depth15min.between_time('9:30','15:45')
        
        depth15min = pd.DataFrame(depth15min)
        depth15min.columns = ["avg_log_depth"]
        depth15min = depth15min.reset_index()
        
        depth15min['date'] = pd.to_datetime(depth15min.date_time.dt.date)
        depth15min['time'] = depth15min.date_time.dt.time
        
        # Nos quedamos con los datos de los dias habiles
        depth15minF = depth15min[depth15min.date.isin(bus_days)]
        
        # Quitamos columnas que ya no necesitamos
        depth15minF = depth15minF.drop(['date','time'],axis=1)
        
        # Pivoteamos los datos para que queden como indice la hora (vertical) y la fecha (horizontal)
        depth15minF = depth15minF.pivot_table(index=[depth15minF.date_time.dt.time, depth15minF.date_time.dt.date]).unstack(1)
        
        # Verificamos esta condicion que hizo Catalina
        depth15minF = depth15minF[depth15minF.sum(1)!=0]
        
        return depth15minF
    
    # Volatilidad
    elif(param_name == "Volatilidad"):
        x1 = stockdata[stockdata.tipo == "TRADE"]
        
        vol15max =  x1.precio.resample("15T").max()
        vol15min =  x1.precio.resample("15T").min()
        logmax   = np.log(vol15max)
        logmin   = np.log(vol15min)
        #Parkinson (1980)
        # daily vol
        vola15min = np.sqrt(25)*np.sqrt(((logmax - logmin)**2)/(4*np.log(2))) 
        #there are 25, 15 min intervals in trading day
        vola15min = vola15min.between_time('9:30','15:45')
        vola15min = vola15min.reset_index()
        
        vola15min['date'] = pd.to_datetime(vola15min.date_time.dt.date)
        vola15min['time'] = vola15min.date_time.dt.time
        
        # Nos quedamos con los datos de los dias habiles
        vola15minF=vola15min[vola15min.date.isin(bus_days)]
        
        # Quitamos las columnas que ya no necesitamos
        vola15minF = vola15minF.drop(['date','time'],axis=1)
        
        # Pivoteamos los datos para que queden como indice la hora (vertical) y la fecha (horizontal)
        vola15minF = vola15minF.pivot_table(index=[vola15minF.date_time.dt.time, vola15minF.date_time.dt.date]).unstack(1)
        
        # Verificamos esta condicion que hizo Catalina
        vola15minF = vola15minF[vola15minF.sum(1)!=0]
        
        return vola15minF
    
    # Spread
    elif(param_name == "Spread"):
        spread15min = stockdata.Quoted_Spread.resample("15T").mean()
        spread15min = spread15min.between_time('9:30','15:45')
        spread15min = spread15min.reset_index()
        
        spread15min['date'] = pd.to_datetime(spread15min.date_time.dt.date)
        spread15min['time'] = spread15min.date_time.dt.time
        
        # Nos quedamos con los datos de los dias habiles
        spread15minF = spread15min[spread15min.date.isin(bus_days)] 
        
        # Quitamos las columnas que ya no necesitamos
        spread15minF=spread15minF.drop(['date','time'],axis=1)
        
        # Pivoteamos los datos para que queden como indice la hora (vertical) y la fecha (horizontal)
        spread15minF = spread15minF.pivot_table(index=[spread15minF.date_time.dt.time, spread15minF.date_time.dt.date]).unstack(1)
        
        # Verificamos esta condicion que hizo Catalina
        spread15minF = spread15minF[spread15minF.sum(1)!=0]
        
        return spread15minF
    
    # Volumen
    elif(param_name == "Volumen"):
        x1 = stockdata[stockdata.tipo == "TRADE"]
        
        vol15min = x1.volumen.resample("15T").sum().fillna(value=0)
        vol15min = vol15min.reset_index()
        
        vol15min['date'] = pd.to_datetime(vol15min.date_time.dt.date)
        vol15min['time'] = vol15min.date_time.dt.time
        
        #select only business days colombia
        vol15minF=vol15min[vol15min.date.isin(bus_days)]
        
        # Quitamos las columnas que ya no necesitamos
        vol15minF=vol15minF.drop(['date','time'],axis=1)
        
        # Pivoteamos los datos para que queden como indice la hora (vertical) y la fecha (horizontal)
        vol15minF = vol15minF.pivot_table(index=[vol15minF.date_time.dt.time, vol15minF.date_time.dt.date]).unstack(1)
        
        # Tomamos el log-volumen
        vol15minP = np.log(vol15minF["volumen"] + 1)
        
        # Verificamos esta condicion que hizo Catalina
        vol15minP = vol15minP[vol15minP.sum(1) != 0]
        
        # Remover ceros
        vol15minP.replace(0.0, np.nan, inplace=True)
        
        return vol15minP
```

```Python
# Funcion que grafica los resultados

def graph(stockdata, stockticker, param_name):
    
    df = get_stock_param(stockdata, stockticker, param_name)
    xx = []
    yy = []
    
    cont = 0 #El contador sera nuestro eje x, pondremos las horas correspondientes sobre esto
    xxq  = []
    yy25 = []
    yy50 = []
    yy75 = []
    
    # Primer ciclo para recorrer las horas (filas) 
    for row in df.iterrows():
        cont += 1
        xxq.append(cont)
        
        yy25.append( np.nanpercentile(row[1].values, 25) )
        yy50.append( np.nanpercentile(row[1].values, 50) )
        yy75.append( np.nanpercentile(row[1].values, 75) )
        
        # Segundo ciclo para recorrer los dias (columnas)
        for i in range(len(row[1])):
            xx.append(cont)
            yy.append(row[1][i])
            
            
    xx = np.array(xx)
    yy = np.array(yy)
    xy = np.vstack([xx, yy])
    # La funcion stats.gaussian_kde() no acepta NaN, entonces aqui los reemplazo por cero 
    # despues de ya haber calculado los cuantiles
    xy[np.isnan(xy)] = 0.0    
    z  = stats.gaussian_kde(xy)(xy) # Este comando colorea los puntos

    idx = z.argsort() # Ordenamos los puntos para que queden los mas rojos encima de los azules
    xx, yy, z = xx[idx], yy[idx], z[idx]
    
    # Ticks eje x
    timelab = ('9:15','9:30','9:45','10:00','10:15','10:30','10:45','11:00', '11:15','11:30','11:45','12:00',
               '12:15','12:30','12:45','13:00','13:15', '13:30','13:45','14:00','14:15','14:30','14:45',
               '15:00','15:15','15:30','15:45','16:00')
    
    # Convertimos a array
    xxq  = np.array(xxq)
    yy25 = np.array(yy25)
    yy50 = np.array(yy50)
    yy75 = np.array(yy75)
    
    # Limites de la figura
    y_min = np.nanmin(yy)
    y_max = np.nanmax(yy)
    delta = 0.05*(y_max-y_min) # Ajuste para los ylim de la grafica
        
    # Grafica
    plt.figure(figsize=(17, 10))
    
    plt.scatter(xx, yy, c=z, s=100, edgecolor='') # Todos los datos
    plt.plot(xxq, yy25, linewidth = 1, c = "g")   # Cuantil 25
    plt.plot(xxq, yy50, linewidth = 2, c = "r")   # Cuantil 50
    plt.plot(xxq, yy75, linewidth = 1, c = "g")   # Cuantil 75
    
    # Titulo y ejes
    plt.title(param_name + " " + stockticker, fontsize=30)
    plt.xlabel("Hora",     fontsize = 20)
    plt.ylabel(param_name, fontsize=20)
    plt.xticks(np.arange(27), timelab, rotation=17)
    plt.colorbar()
    
    plt.ylim((y_min-delta, y_max+delta))
    plt.show()
```
