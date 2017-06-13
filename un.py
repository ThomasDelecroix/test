# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:12:21 2017

@author: thomasdelecroix
"""

#Load in some libraries to handle the web page requests and the web page parsing...
import requests
from   bs4         import BeautifulSoup
from   urlparse    import parse_qs
from   io          import BytesIO
import zipfile
import pandas      as pd
import numpy       as np
import os
from   scipy       import stats
from   scipy.stats import chi2_contingency
from   scipy.stats import f_oneway
from   collections import Counter

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

workingDirectory = '/Users/thomasdelecroix/Desktop/Lesaffre/projet'
os.chdir(workingDirectory)
ZIPdir  = './data/zip/'
CSVdir  = './data/csv/'
SDMXdir = './data/SDMX/'

#The scraper will be limited to just the first results page...
def searchUNdata(q):
    ''' Run a search on the UN data website and scrape the results '''
    
    params={'q':q}
    url='http://data.un.org/Search.aspx'

    response = requests.get(url,params=params)

    soup     = BeautifulSoup(response.content, "html5lib")

    results={}

    #Get the list of results
    searchresults=soup.findAll('div',{'class':'Result'})
    
    #For each result, parse out the name of the dataset, the datamart ID and the data filter ID
    for result in searchresults:
        h2=result.find('h2')
        #We can find everything we need in the <a> tag...
        a=h2.find('a')
        p=parse_qs(a.attrs['href'])
        if ('d' in p):
            results[a.text]=(p['d'][0],p['f'][0])
        else:
            results[a.text]=(p['id'][0],)

    return results

#A couple of helper functions to let us display the results

def printResults(results):
    ''' Nicely print the search results '''
    
    for result in results.keys():
        print(result)


def unDataSearch(q):
    ''' Simple function to take a search phrase, run the search on the UN data site, and print and return the results. '''
    
    results=searchUNdata(q)
    #printResults(results)
    return results

#Just in case - a helper routine for working with the search results data
def search(d, substr):
    ''' Partial string match search within dict key names '''
    #via http://stackoverflow.com/a/10796050/454773
    
    result = []
    for key in d:
        if substr.lower() in key.lower():
            result.append((key, d[key])) 

    return result


def getUNdata(undataSearchResults,dataset):
    ''' Download a named dataset from the UN Data website and load it into a pandas dataframe '''
    res = undataSearchResults[dataset]
    if len(undataSearchResults[dataset])==1:
        sourceID, = res
        url      = 'http://data.un.org/Handlers/DocumentDownloadHandler.ashx?id='+sourceID+'&t=bin'
    else:
        datamartID,seriesRowID = undataSearchResults[dataset]
        url                    = 'http://data.un.org/Handlers/DownloadHandler.ashx?DataFilter='+seriesRowID+'&DataMartId='+datamartID+'&Format=csv'

    r = requests.get(url)
    
    
    s = BytesIO(r.content)
    z = zipfile.ZipFile(s)

    df=pd.read_csv( BytesIO( z.read( z.namelist()[0] ) ), error_bad_lines=False).dropna(axis=1,how='all')
    return df

def dropFootnotes(df):
    try:  
        return df[:((df[df.columns[0]]=='footnoteSeqID').tolist().index(True)-1)]
    except ValueError:
        return df

def getUNdata2(undataSearchResults, dataset, footnotes=False):
    df=getUNdata(undataSearchResults, dataset)
    if footnotes:
        return df
    return dropFootnotes(df)

def loadUNdata(dataset):
    print dataset[1]
    escapedKeywords = dataset[0].replace('"','\\"')
    searchRes       = unDataSearch(escapedKeywords)
    
    if (len(searchRes) == 0):
        searchRes = unDataSearch('"' + escapedKeywords + '"')
    return getUNdata2(searchRes, dataset[1])

# Signifiance :
## Continuous vs. Nominal: ANOVA
## Nominal    vs. Nominal: X2
# Strenght
## Continuous vs. Nominal: intraclass correlation
## Nominal    vs. Nominal: Cramer's V

def chisq_of_df_cols(df, c1, c2):
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    confusion_matrix = ctsum.fillna(0)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    return chi2, p, dof, expected, confusion_matrix

def cramers_stat(df, c1, c2):
    chi2, p, dof, expected, confusion_matrix = chisq_of_df_cols(df, c1, c2)
    n = confusion_matrix.values.sum()
    return round(np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1))),2)

def anova_test(df, group_var, num_var):
    #df.boxplot(num_var, by=group_var, figsize=(12, 8))
 
    grps = pd.unique(df[group_var].values)
    d_data = {grp:df[num_var][df[group_var] == grp] for grp in grps}
     
    F, p = f_oneway(*d_data.values())
    return p

def safeFilename(unsafeFilename):
    for ch in ['<','>', ":", '"', '"', "/", "\\", "|", "?", "*"]:
        if ch in unsafeFilename:
            unsafeFilename = unsafeFilename.replace(ch," ")
    return unsafeFilename

def multiple_csv(dfList, dfNamesList):
    i=0
    for df in dfNamesList:
        dfList[i].to_csv(CSVdir + safeFilename(dfNamesList[i][1]) + ".csv")
        i+=1
    return True

def missingValues(df):
    return df.isnull().sum(axis=0)

def frequenciesCount(df,case='Nom'):
    if case.upper()   == 'NOM':
        df_cat = df.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    elif case.upper() == 'NUM':
        df_cat = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    elif case.upper() == 'ALL':
        df_cat = df
    
    frequencyDict = {}    
    for col in df_cat.columns:
        thisFreq           = pd.concat([df_cat[col].value_counts(), df_cat[col].value_counts()/df_cat[col].value_counts().sum()], axis=1)
        thisFreq.columns   = [thisFreq.columns.values[0]+"_cnt", thisFreq.columns.values[1]+"_freq"]
        frequencyDict[col] = thisFreq
    return frequencyDict

def frequenciesCountBy(df, byList, case='Nom'):
    if case.upper()   == 'NOM':
        df_cat = df.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    elif case.upper() == 'NUM':
        df_cat = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    elif case.upper() == 'ALL':
        df_cat = df
    
    frequencyDict = {}    
    for col in listSubstration(df_cat.columns.tolist(), byList):
        cnt                = df[set().union(*[df_cat.columns.tolist(), byList])].groupby[byList].agg('count')
        freq               = cnt/df[set().union(*[df_cat.columns.tolist(), byList])].value_counts().sum()
        thisFreq           = pd.concat([cnt, freq], axis=1)
        thisFreq.columns   = [thisFreq.columns.values[0] + "_cnt", thisFreq.columns.values[1] + "_freq"]
        frequencyDict[col] = thisFreq
    return frequencyDict

def histValues(df):
    df_num   = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    histList = {}  
    for col in df_num.columns: 
        df_num[col].value_counts()
        uniq = df_num[col].unique()
        n    = len(uniq)
        count, division = np.histogram(df["Year"], bins=len(df["Year"].unique()))
        histList[col] = { "Nombre de valeurs uniques" : n, "Valeurs uniques" : uniq, "bornes histograme" : division, "Compte" : count}
    return histList

def anova_test2(df, group, var):
    k = len(pd.unique(df[group]))  # number of conditions
    N = len(df.values)  # conditions times participants
    n = df.groupby(group).size()[0] #Participants in each condition
    DFbetween     = k - 1
    DFwithin      = N - k
    DFtotal       = N - 1
    SSbetween     = (sum(df.groupby(group).sum()[var]**2)/n) - (df[var].sum()**2)/N
    sum_y_squared = sum([value**2 for value in df[var].values])
    SSwithin      = sum_y_squared - sum(df.groupby(group).sum()[var]**2)/n
    SStotal       = sum_y_squared - (df[var].sum()**2)/N
    MSbetween     = SSbetween/DFbetween
    MSwithin      = SSwithin/DFwithin
    F             = MSbetween/MSwithin
    p             = stats.f.sf(F, DFbetween, DFwithin)
    eta_sqrd      = SSbetween/SStotal
    om_sqrd       = (SSbetween - (DFbetween * MSwithin))/(SStotal + MSwithin)
    return { 'DFbetween' : DFbetween, 'DFwithin' : DFwithin, 'DFtotal' : DFtotal, 'SSbetween' : SSbetween, 'sum_y_squared' : sum_y_squared, 'SSwithin' : SSwithin, 'SStotal' : SStotal, 'MSbetween' : MSbetween, 'MSwithin' : MSwithin, 'F' : F, 'p' : p, 'eta_sqrd' : eta_sqrd, 'om_sqrd' : om_sqrd } 

def dropEmptyRows(df, threeshold):
    df = df.copy(deep=True)
    df = df[df.isnull().sum(axis=1)/df.shape[1]<threeshold]
    return df

def toEncodeUTF8(df, toEncodeList):
    df = df.copy(deep=True)
    for col in df.columns:
        if (col in toEncodeList):
            df[col] = df[col].str.encode('utf-8')
    return df

def castToFloat(df, toCastList):
    df = df.copy(deep=True)
    for col in df.columns:
        if (col in toCastList):
            df[col] = df[col].convert_objects(convert_numeric=True)
    return df

def castToInt(df, toCastList):
    df = df.copy(deep=True)
    for col in df.columns:
        if (col in toCastList):
            df[col] = df[col].astype(int, errors='ignore')
    return df

def castToCaterogical(df, toCategoricalList, order, modalityList=False):
    df = df.copy(deep=True)
    for col in df.columns:
        if (col in toCategoricalList):
            if (order==True):
                df[col] = df[col].astype('category', categories=modalityList, ordered=True)
            else:
                df[col] = df[col].astype('category', ordered=False)
    return df

def autoCat(df):
    df = df.copy(deep=True)
    objectList = df.select_dtypes(['object']).columns.tolist()
    toCatList = []
    for item in objectList:
        if df[item].nunique()<=10:
            toCatList.append(item)
    for col in df.columns:
        if (col in toCatList):
            df[col] = df[col].astype('category')
    return df

def colToDrop(df, dropList):
    df = df.copy(deep=True)
    for col in df.columns:
        if (col in dropList):
            df = df.drop(col, axis = 1)
    return df

def dropConstant(df):
    df = df.copy(deep=True)
    for col in df.columns:
        if (df[col].nunique()==1):
            df = df.drop(col, axis = 1)
    return df

def preprocess(dfList, toDropList, toIntList, toFloatList, toUTF8List):
    prepocessedList = []
    i = 0
    for df0 in dfList:
        df = df0.copy(deep=True)
        df = colToDrop(df, toDropList)
        df = dropEmptyRows(df, 0.7)
        df = dropConstant(df)
        df = castToInt(df, toIntList)
        df = castToFloat(df, toFloatList)
        df = toEncodeUTF8(df, toUTF8List)
        prepocessedList.append(df)
        i=i+1
    return prepocessedList

def listSubstration(firstList, secondList):
    return [item for item in firstList if item not in secondList]

def keepMaxBy(df, maxVar):
    df = df.copy(deep=True)
    allExceptMaxed = listSubstration(df.columns.tolist(), [maxVar])
    idx            = df.groupby(allExceptMaxed)[maxVar].transform(max) == df[maxVar]
    newDf          = df[idx].drop(maxVar, axis=1)
    return newDf                

def greatPivot(df):
    allColList = df.columns.tolist()
    
    potentialIndexList = ['Country or Area', 'Reference Area',  'Time Period',  'Year']
    potentialValueList = [ 'Value',  'Quantity',  'Trade (USD)']
    
    indexList = []
    valueList = []
    for col in allColList:
        if   (col in potentialIndexList):
            indexList.append(col)
        elif (col in potentialValueList):
            valueList.append(col)
    
    colList    = listSubstration(listSubstration(allColList, valueList), indexList)
    dfPivoted  = pd.pivot_table(df, values=valueList, index=indexList, columns=colList, aggfunc=np.mean)
    n_org = df[valueList].count().sum()
    n_piv = dfPivoted.count().sum()
    s_org = df[valueList].values.sum()
    s_piv = dfPivoted.values.sum()
    dfPivoted = dfPivoted.reset_index()
    case = ""
    if ((n_org != n_piv) or (s_org != s_piv)):
        if (n_org != n_piv):
            if (n_org > n_piv):
                case = case + "Deperdition apres transf"
            else:
                case = case + "Doublonnage apres transf"
        else:
            case = case + "nombre identique mais "
        if (s_org != s_piv):
            if (case != ""):
                case = case + " "
            if (s_org > s_piv):
                case = case + "Diminution   apres transf"
            else:
                case = case + "Augmentation apres transf"
        else:
            case = case + "somme identique"
    else:
        case = "OK"
    print case
    return dfPivoted  

#datasets = dict(zip([x[0] for x in datasetNamesList], map(loadUNdata,datasetNamesList)))
# List of search and files associated to download
datasetNamesList = [
["Population by sex and urban/rural residence","Population by sex and urban/rural residence"],
["Women's share of labour force","Women's share of labour force"],
["Poverty headcount ratio at national poverty lines (% of population)","Poverty headcount ratio at national poverty lines (% of population)"],
["Poverty gap ratio at $1.25 a day (PPP), percentage","Poverty gap ratio at $1.25 a day (PPP), percentage"],
["Per capita GDP at current prices - US dollars","Per capita GDP at current prices - US dollars"],
["Proportion of employed population below the international poverty line of US$1.90 per day (the working poor)","Proportion of employed population below the international poverty line of US$1.90 per day (the working poor)"],
["Proportion of population below the international poverty line of US$1.90 per day","Proportion of population below the international poverty line of US$1.90 per day"],
["GDP","GDP by Type of Expenditure at current prices - US dollars"],
["Growth rate of real GDP per capita","Growth rate of real GDP per capita"],
["Inflation, GDP deflator (annual %)","Inflation, GDP deflator (annual %)"],
["Agriculture, value added (% of GDP)","Agriculture, value added (% of GDP)"],
["Revenue, excluding grants (% of GDP)","Revenue, excluding grants (% of GDP)"],
#["Human Development Index trends, 1990–2014",u'Human Development Index trends, 1990\xe2\u20ac\u201c2014'],
#["Human Development Index and its components","Human Development Index and its components"],
["Consumer prices, food indices (2000=100)","Consumer prices, food indices (2000=100)"],
["Wheat","Wheat"],
["Maize","Maize"],
["Wheat or meslin flour","Wheat or meslin flour"],
["Maize (corn) flour","Maize (corn) flour"],
["Trade of goods , US$, HS 1992, 10 Cereals","Trade of goods , US$, HS 1992, 10 Cereals"],
["11 Milling products","Trade of goods , US$, HS 1992, 11 Milling products, malt, starches, inulin, wheat glute"],
["Crude petroleum","Crude petroleum"],
["Total Electricity","Total Electricity"],
["Improved water source (% of population with access)","Improved water source (% of population with access)"],
["Table 3.2 Individual consumption expenditure of households, NPISHs, and general government at current prices","Table 3.2 Individual consumption expenditure of households, NPISHs, and general government at current prices"],
["Exchange rates","Exchange rates"]
]

toDropList = [
"Value Footnotes",
"Value Footnotes.1",
"Source",
"Subgroup",
"Unit of measurement",
"Element Code",
"Quantity Name",
"Fiscal Year Type",
"SNA93 Item Code",
"SNA93 Table Code",
"Description",
"OID"
]

toIntList = [
"Year"
]

toFloatList = [
"Value"
]

toUTF8List = [
"Currency", 
"Country or Area", 
"Commodity - Transaction"
]

joinKeys = [
'Country or Area',
'Year'
]

rangeKeys = ['Reference Area','Time Period']

oldRecordType = [
"Estimate - de facto",
"Estimate - de jure",
"Census - de facto - complete tabulation",
"Census - de jure - complete tabulation",
"Sample survey - de facto",
"Record type not defined/applicable",
"Sample survey - de jure",
"Census - de facto - sample tabulation",
"Census - de jure - sample tabulation"
]

newRecordType = [
5,
6,
8,
9,
7,
1,
2,
3,
4
]



oldReliability = [
"Final figure, complete",
"Final figure, incomplete/questionable reliability",
"Provisional figure",
"Provisional figure with questionable reliability",
"Other estimate"
]

newReliability = [
5,
3,
4,
2,
1
]

datasetsList        = map(loadUNdata, datasetNamesList)
preprocessedList    = preprocess(datasetsList, toDropList, toIntList, toFloatList, toUTF8List)
# keep only the latest source year for the population dataset
preprocessedList[0] = keepMaxBy(preprocessedList[0],'Source Year')
# recoding reliability and Record Type to apply max
test = pd.concat([preprocessedList[0], preprocessedList[0]["Record Type"].map(dict(zip(oldRecordType, newRecordType)))], axis = 1, ignore_index=True)
test2 = test["Reliability"].map(dict(zip(oldReliability, newReliability)))
freq(preprocessedList[0]["Year"])
pivotedList         = map(greatPivot, preprocessedList)
#greatPivot(preprocessedList[3])
freq = frequenciesCount(preprocessedList[0], 'NUM')
freqBy

freqlist          = map(frequenciesCount, dataFrameList)

####### Transformation #####



oList = dataFrameList[0].select_dtypes(['object']).columns.tolist()


prout = pd.concat([
        dataFrameList[0].select_dtypes([], ['object']),
        dataFrameList[0].select_dtypes(['object']).apply(pd.Series.astype, dtype = 'str')
        ], axis=1).reindex_axis(dataFrameList[0].columns, axis=1)
prout["Year"]        = prout["Year"].astype(int)
prout["Source Year"] = prout["Source Year"].astype(int)
prout = prout.drop(['Value Footnotes', 'Value Footnotes.1'], axis=1)

colList = prout.columns.values.tolist()
allExceptSource = [item for item in colList if item not in ['Source Year', 'Value']]
prout = prout[(prout.Year >= 2007)]
# selection by dtypes
# prout.select_dtypes(['int64'],[]).columns.values.tolist()

idx       = prout.groupby(allExceptSource)['Source Year'].transform(max) == prout['Source Year']
prout1    = prout[idx].drop(['Source Year'], axis=1)
realIndex = ['Country or Area', 'Year', 'Value']
# pie chart 


'Final figure, complete'
'Final figure, incomplete/questionable reliability'
'Provisional figure'
'Provisional figure with questionable reliability'


'Estimate - de jure'
'Estimate - de facto'
'Census - de jure - complete tabulation'
'Census - de facto - complete tabulation'
'Sample survey - de facto'
'Census - de jure - sample tabulation'
'Record type not defined/applicable'
'Sample survey - de jure'


prout['Reliability'].value_counts().sort_values(ascending=False).plot.pie(figsize=(3, 3))
prout['Reliability'].value_counts().sort_values(ascending=False)
fiability = pd.DataFrame(prout[['Country or Area', 'Reliability']].groupby('Country or Area')['Reliability'].value_counts().sort_values(ascending=False)).unstack(-1).fillna(0)
fiability['Total Reliability']=pd.DataFrame(prout[['Country or Area', 'Reliability']].groupby('Country or Area')['Reliability'].value_counts().sort_values(ascending=False)).unstack(-1).fillna(0).sum(axis=1)

fiability['Reliable']= fiability.iloc[:,0]+fiability.iloc[:,2]
fiability['Non Reliabile']= fiability.iloc[:,1]+fiability.iloc[:,3]
fiability['Reliability percent'] = fiability['Reliable']/fiability['Total Reliability']
fiability = fiability.sort_values('Reliability percent', ascending=False)
tot=pd.DataFrame([fiability.iloc[:,0]+fiability.iloc[:,2], fiability.iloc[:,1]+fiability.iloc[:,3]], columns=['Fiable','Non Fiable'])

colList   = prout1.columns.values.tolist()
cols      = [item for item in colList if item not in realIndex]
prout2    = pd.pivot_table(prout1, values='Value', index=['Country or Area', 'Year'], columns=cols, aggfunc=np.mean)
prout2 = prout2.reset_index()
df = pd.Series((len(prout2.index)-prout2.count())/prout2.count())
df = pd.Series(prout2.isnull().sum()/prout2.shape[0]).sort_values(ascending=False)
prout1['Value'].count() - prout2.count().sum()
df = pd.DataFrame(prout1.groupby(['Country or Area', 'Year','Area', 'Sex', 'Record Type', 'Reliability'], as_index=False)['Value'].count()).sort_values('Value', ascending=False)
df=df[df['Value']>1]
fiability.iloc[:,7].pie()
pr = prout[(prout['Country or Area']=='Croatia') & (prout1['Year']==2014) & (prout1['Area']=='Total') & (prout1['Sex']=='Male') & (prout1['Record Type'] == 'Estimate - de jure') & (prout1['Reliability'] == 'Final figure, complete')]

prout2 = prout.groupby(['Country or Area', 'Year','Area', 'Sex', 'Record Type', 'Reliability'], as_index=False)['Value'].sum()
prout2 = prout2.unstack(['Area', 'Sex', 'Record Type', 'Reliability', 'Source Year'])
prout2.reset_index()
prout2.columns.values
#v = cramers_stat(prout,'Country or Area', 'Reliability')
#p = anova_test(prout,'Country or Area', 'Value')
#dliste = [dataFrameList[i].describe(include='all',percentiles=[0, 0.25, 0.5, 0.75, 1]) for i in range(len(dataFrameList))]

multiple_csv(dataFrameList, datasetNamesList)


flatColumnList = [item for sublist in columnList for item in sublist]
flatUniqColumnList = list(set(flatColumnList))
keepCol = ['Country or Area', 'Year']
colToRename = [item for item in flatUniqColumnList if item not in keepCol]
i=1
columnList = []
for df in dataFrameList:
    for colName in colToRename:
        if colName in df.columns.values:
            df.rename(columns={colName: 'T'+str(i)+'-'+colName}, inplace=True)
    columnList.append(df.columns.values)
    i+=1
flatColumnList = [item for sublist in columnList for item in sublist]
counts = Counter(flatColumnList)
print(counts)

cl=pd.DataFrame(columnList)
# to do : crosstable (Pays+Annee) X (toutes les autres) X dernière colonne
# PCA + Classif
# correlations