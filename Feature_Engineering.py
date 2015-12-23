import pandas as pd

#read into data
data=pd.read_csv(r'/home/zyybear/Desktop/UNH/RS_Kaggle/train/Complete_Train.csv')

#change salesmeans to 0 if store is closed
data.ix[data.Open==0,"SalesMean"]=0
data.ix[data.Open==0,"CustomerMean"]=0
data.ix[data.Open==0,"PerCustomerSales"]=0

data.ix[(data.Open==1) & (data.Sales==0),"Open"]=0

# dealing with competition open since data
#fill missing value with 0 first
data.CompetitionOpenSinceMonth.fillna(0, inplace=True)
data.CompetitionOpenSinceYear.fillna(0, inplace=True)
#initilize to 0 for compt since month
data['CompetSinceMonths']=-99999
data.ix[(data.CompetitionOpenSinceMonth != 0),'CompetSinceMonths']=(data['Year']-data['CompetitionOpenSinceYear'])*12+(data['Month']-data['CompetitionOpenSinceMonth'])
data.ix[(data.CompetitionOpenSinceMonth == 0),'CompetSinceMonths']=-99999

CompetSinceMonthsMean=data.ix[data.CompetSinceMonths>0].CompetSinceMonths.mean()
print CompetSinceMonthsMean
#fill the missing competition open information with mean value
data.ix[data.CompetSinceMonths==-99999].CompetSinceMonths.fillna(CompetSinceMonthsMean, inplace=True)

data.ix[data.CompetSinceMonths<0,'CompetSinceMonths']=0
data.ix[data.Store==815,"CompetSinceMonths"]=CompetSinceMonthsMean
data.ix[data.Store==146,"CompetSinceMonths"]=CompetSinceMonthsMean

#create new columns to indicate whether a store has available competition.
#1 means has competition,0 means no competition

#assume all sotres have competition since most of them have competition distance available
data['hasCompe']=1


#the stores that have missing compet distance do not have compet
#fill the missing cmpt distance with "-99"
data.CompetitionDistance.fillna(-99,inplace=True)

#assign "hasCompe" to 0 if it does not have any cmpt
data.ix[data.CompetitionDistance==-99,'hasCompe']=0

#assign "CompetSinceMonths" to 0 if it does not have any cmpt
data.ix[data.CompetitionDistance==-99,'CompetSinceMonths']=0

#assign the -99 compet distance with max compet distance
data.ix[data.CompetitionDistance==-99,'CompetitionDistance']=75860.0

data.Promo2SinceWeek.fillna(0, inplace=True)
data.Promo2SinceYear.fillna(0, inplace=True)

#dealing with promo2 columns

#initialize to 0 for promo2Length columns
data['promo2Length']=0

data.ix[data.Promo2SinceYear!=0,'promo2Length']=(data['Year']-data['Promo2SinceYear'])*52+(data['Month']*4-data['Promo2SinceWeek'])
data.ix[data.promo2Length<0,'promo2Length']=0

data['WeekOfYear'] = pd.DatetimeIndex(data['Date']).week

data.drop(['Promo2SinceWeek','Promo2SinceYear','CompetitionOpenSinceMonth',
'CompetitionOpenSinceYear'],axis=1,inplace=True)

#get the season data
data.ix[(data.Month >= 9) & (data.Month <= 11)  ,'Season'] = 'Fall'
data.ix[(data.Month >= 12) | (data.Month <= 2)  ,'Season'] = 'Winter'
data.ix[(data.Month >= 2) & (data.Month <= 5)  ,'Season'] = 'Spring'
data.ix[(data.Month >= 6) & (data.Month <= 8)  ,'Season'] = 'Summer'

#if promo length==0,then Promo2 should be 0 as well
data.ix[data.promo2Length==0,'Promo2']=0

#if competSinceMonths is 0, then hasCompe should be 0
data.ix[data.CompetSinceMonths==0,'hasCompe']=0

#add IsPromoMonth columns
#inilize to 0
data["IsPromoMonth"]=0
data.PromoInterval.fillna("No", inplace=True)
PromoDict={"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
            "Jul":7,"Aug":8,"Sept":9,"Oct":10,"Nov":11,"Dec":12}
data.ix[data.PromoInterval=='No',"IsPromoMonth"]=0

for ele in data.PromoInterval.unique():
    if ele!="No":
       for month in ele.split(','):
           MonthInt=PromoDict[month]
           data.ix[(data.Month==MonthInt) & (data.PromoInterval==ele)
                   &(data.promo2Length!=0),"IsPromoMonth"]=1

#add columns that show which part is in the month, like first 10 days, middle 10 days
# or last 10 days
data['whereInMonth']='null'
data.ix[data.DayInMonth<=10,'whereInMonth']='first'
data.ix[(data.DayInMonth>10) & (data.DayInMonth<=20),'whereInMonth']='middle'
data.ix[data.DayInMonth>20,'whereInMonth']='end'     

#now transform categorical value to numeric value
#first convert them to string object for future get dummies
for col in ['DayOfWeek','Promo','SchoolHoliday','Open','StateHoliday','DayInMonth','Month'
            ,'Promo','Promo2','hasCompe','Year'
           ]:
    data[col]=data[col].astype(str)

categorical_variables=['DayOfWeek','Promo','SchoolHoliday','State',
                       'Open','StateHoliday','DayInMonth','hasCompe',
                        'StoreType','Assortment','Promo2','Month','Season','Year',
                      'whereInMonth']

regular_variables=['Store','Date','Sales','Customers','CompetitionDistance','SalesMean',
                    'CompetSinceMonths','promo2Length','CustomerMean','PerCustomerSales',
                  "IsPromoMonth",'WeekOfYear']

dummy_table=pd.DataFrame()
for var in categorical_variables:
    dummy_table=pd.concat([dummy_table,pd.get_dummies(data[var], prefix=var)], axis=1) 
    
data_dummy=pd.concat([dummy_table,data[regular_variables]],axis=1)

#whether open on Holiday
data_dummy["OpenOnHoliday"]=0
data_dummy.ix[(data_dummy.Open_1==1) & (data_dummy.StateHoliday_a==1),"OpenOnHoliday"]=1
data_dummy.ix[(data_dummy.Open_1==1) & (data_dummy.StateHoliday_b==1),"OpenOnHoliday"]=1
data_dummy.ix[(data_dummy.Open_1==1) & (data_dummy.StateHoliday_c==1),"OpenOnHoliday"]=1

#if it open on 23 dec, consider it as open on holiday.
data_dummy.ix[(data_dummy.DayInMonth_23==1) & (data_dummy.Month_12==1) 
              & (data_dummy.Open_1==1),"OpenOnHoliday"]=1

#if it open on Sunday, consider it as open on holiday.
data_dummy.ix[(data_dummy.Open_1==1) & (data_dummy.DayOfWeek_7==1),"OpenOnHoliday"]=1

data_dummy.to_csv('train_FE0.csv', index=False, sep=',')
