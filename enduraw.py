import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# extraire et mettre en series pandas les données
file_path = 'Norseman_all_years.xlsx'
data = pd.read_excel(file_path)

#Supprimer les données inutiles et les lignes où des données manquent
data.drop(['Club', 'Swim distance', 'Bike distance', 'Run distance'], axis=1, inplace=True)
data = data.dropna()

#eliminer les annees mixtes (pas d'arrivée au sommet) 
ann = [2005,2007,2023]
for a in ann:
    data = data[data['Year'] != a]


#Convertir les temps en numériques (heures)
temps = ['Swim time', 'Time T1', 'Time T2','Cycle time', 'Run time', 'Total time']
for t in temps:
    data[t] = data[t].dt.total_seconds()/3600


#eliminer les données abérantes par rapport aux barrière horaires
data = data[data['Swim time']<2.5]
data = data[data['Cycle time']<12]

data = data.reset_index(drop=True)

#Ajouter les classments intermédiaires par années
data_swim = data.sort_values(['Year', 'Swim time'], ascending=[True,True])
data['Rank swim'] = data_swim.groupby('Year')['Swim time'].rank(method='min',pct=True)*100

#Les données des transitions comportent beaucoup d'erreur (minutes transformés en heures), on ne les prendra pas en compte et elles restent négligeables vu la longueur de la course
data['Time after cycle']= data['Swim time']+data['Cycle time']

data_run = data.sort_values(['Year', 'Time after cycle'], ascending=[True,True])
data['Rank after cycle'] = data_run.groupby('Year')['Time after cycle'].rank(method='min',pct=True)*100

data_fin = data.sort_values(['Year', 'Total time'], ascending=[True,True])
data['Rank final'] = data_run.groupby('Year')['Total time'].rank(method='min',pct=True)*100

data_cycle = data.sort_values(['Year', 'Cycle time'], ascending=[True,True])
data['Rank cycle'] = data_cycle.groupby('Year')['Cycle time'].rank(method='min',pct=True)*100

data_runLeg = data.sort_values(['Year', 'Run time'], ascending=[True,True])
data['Rank run'] = data_runLeg.groupby('Year')['Run time'].rank(method='min', pct=True)*100

data['Gain cycle'] = data['Rank after cycle']-data['Rank swim']

data['Gain run'] = data['Rank final']-data['Rank after cycle']



#Différencier les top finishers des autres
top_finish = data[data['Finish at']=='Gaustatoppen']
down_finish = data[data['T-shirt']=='White']


#Histogramme des temps totaux selon la cétogrie de finishers
plt.figure(figsize=(9,5))
serie1 = top_finish['Total time']
serie2 = down_finish['Total time']
top_median = np.average(top_finish['Total time'])
down_median = np.average(down_finish['Total time'])
plt.axvline(top_median, color='blue', linestyle='dashed', label = 'Black tshirt finishers average time')
plt.axvline(down_median, color='red', linestyle='dashed', label='White tshirt finishers average time')
plt.hist([serie2,serie1], bins=100, alpha=0.5, color=['red','blue'], label=['White tshirt finishers','Black tshirt finishers'] ,stacked=True)
plt.xlabel('Time in hours')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#Afficher les temps min et max des concurrents finis avec le black tshirt 
toptime = top_finish.groupby('Year')['Total time'].min()
maxtime = top_finish.groupby('Year')['Total time'].max()
print(toptime.mean(), maxtime.mean())

maxtime.plot(kind='bar', label='Maximum time', stacked=True, color='red',alpha=0.8)
toptime.plot(kind='bar', label='Minimum time', stacked=True, color='blue')

plt.title('Black t-shirt finishers time')
plt.ylabel('Time (in hours)')
plt.xlabel('')
plt.legend()
plt.show()

#calcul du coeficient de variation des temps maximum des blackfinishers par année
cv = (maxtime - maxtime.shift(1)) / maxtime.shift(1)
moy_cv = cv.mean()
print(moy_cv)


#Régression entre position a la fin de la natation et position finale
gain_cycle_std = np.std(top_finish['Gain cycle'])
gain_run_std = np.std(down_finish['Gain run'])
data['Gain cycle'].plot.hist(bins=100, alpha=0.5, color='red', label=f'Gain cycle (SD = {gain_cycle_std:.2f})')
data['Gain run'].plot.hist(bins=100, alpha=0.5, color='blue', label=f'Gain run (SD = {gain_run_std:.2f})')
plt.xlabel('Positions gained(-) / lost(+)')
plt.legend()
plt.show()


#Régression entre les temps de chaque discipline sur le ranking final
plt.subplot(1,3,1)
plt.scatter(top_finish['Rank swim'],top_finish['Rank final'], color='blue', alpha=0.7,s=4, label='Black tshirt finisher')
plt.scatter(down_finish['Rank swim'],down_finish['Rank final'], color='red', alpha=0.7,s=4, label='White tshirt finisher')

a = data[['Rank swim']]
b = data['Rank final']
model = LinearRegression()
model.fit(a, b)
b_pred = model.predict(a)
data['b_pred'] = b_pred
r2 = r2_score(b, b_pred)
plt.plot(data['Rank swim'], data['b_pred'], color='black', label=f'Regression line (R² = {r2:.2f})')
plt.title('Swim leg',fontsize=14)
plt.xlabel('Rank on swim leg (%)')
plt.ylabel('Final rank (%)')
plt.legend()


plt.subplot(1,3,2)
plt.scatter(top_finish['Rank cycle'],top_finish['Rank final'], color='blue', alpha=0.7,s=4, label='Black tshirt finisher')
plt.scatter(down_finish['Rank cycle'],down_finish['Rank final'], color='red', alpha=0.7,s=4, label='White tshirt finisher')

a = data[['Rank cycle']]
b = data['Rank final']
model = LinearRegression()
model.fit(a, b)
b_pred = model.predict(a)
data['b_pred'] = b_pred
r2 = r2_score(b, b_pred)
plt.plot(data['Rank cycle'], data['b_pred'], color='black', label=f'Regression line (R² = {r2:.2f})')
plt.title('Bike leg', fontsize=14)
plt.xlabel('Rank on bike leg (%)')
plt.ylabel('Final rank (%)')
plt.legend()


plt.subplot(1,3,3)
plt.scatter(top_finish['Rank run'],top_finish['Rank final'], color='blue', alpha=0.7,s=4, label='Black tshirt finisher')
plt.scatter(down_finish['Rank run'],down_finish['Rank final'], color='red', alpha=0.7,s=4, label='White tshirt finisher')

a = data[['Rank run']]
b = data['Rank final']
model = LinearRegression()
model.fit(a, b)
b_pred = model.predict(a)
data['b_pred'] = b_pred
r2 = r2_score(b, b_pred)
plt.plot(data['Rank run'], data['b_pred'], color='black', label=f'Regression line (R² = {r2:.2f})')
plt.title('Run leg', fontsize=14)
plt.xlabel('Rank on run leg (%)')
plt.ylabel('Final rank (%)')
plt.legend()

plt.show()


#Observation du parcours des athlètes du top 5%
top = data[data['Rank final'] <=5]

x = [1, 2,3]

for i in top.index:
    y = [
        top.loc[i, 'Rank swim'],
        top.loc[i, 'Rank after cycle'],
        top.loc[i, 'Rank final']
    ]
    plt.plot(x, y, marker='o',color='black', alpha=0.5, linewidth=0.2, markersize=2)

plt.title('Athlete Rankings Across Stages', fontsize=14)
plt.ylabel('Rank (%)', fontsize=12)
plt.xticks(x, ['After swim','After cycle', 'Final'], fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis() 
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


data['Tranche'] = (data['Rank after cycle'] // 5) * 5

# Calculer la proportion de "Finish at" Gaustatoppen pour chaque tranche
proportion = data[data['Finish at'] == 'Gaustatoppen'].groupby('Tranche')['Finish at'].count() / data.groupby('Tranche')['Finish at'].count() * 100
proportion = proportion.dropna()

#Mise en graphique de la proportion
proportion.plot(kind='line')
plt.axvline(x=60, color='green', linestyle='--')
plt.xlabel('Ranking after cycling (%)')
plt.ylabel('Proportion of black t-shirt finishers (%)')
plt.show()

#Temps moyen de la proportion des 60%
bike60 = data[(data['Rank cycle'] >= 59) & (data['Rank cycle'] <= 61)]
print("Swim = ")
print(bike60['Swim time'].mean())
print("Bike = ")
print(bike60['Cycle time'].mean())
print("Run = ")
print(bike60['Run time'].mean())



#export des données complètes dans un nouveau fichier csv
#data.to_csv('updated_norseman.csv', index=False, encoding='utf-8')



## EXPLORATOIRE

#SwimT = top_finish.groupby('Year')['Swim time'].min()
#SwimD = down_finish.groupby('Year')['Swim time'].max()
#T1T = top_finish.groupby('Year')['Time T1'].min()
#T1D = down_finish.groupby('Year')['Time T1'].max()
#BikeT = top_finish.groupby('Year')['Cycle time'].min()
#BikeD = down_finish.groupby('Year')['Cycle time'].max()
#T2T = top_finish.groupby('Year')['Time T2'].min()
#T2D = down_finish.groupby('Year')['Time T2'].max()
#RunT = top_finish.groupby('Year')['Run time'].min()
#RunD = down_finish.groupby('Year')['Run time'].max()
#
#SwimT.plot(kind='bar', label='top swim')
#plt.legend()
#plt.show()
#SwimD.plot(kind='bar', label='down swim')
#plt.legend()
#plt.show()
#T1T.plot(kind='bar', label='top t1')
#plt.legend()
#plt.show()
#T1D.plot(kind='bar', label='down t1')
#plt.legend()
#plt.show()
#BikeT.plot(kind='bar', label='top bike')
#plt.legend()
#plt.show()
#BikeD.plot(kind='bar', label='down bike')
#plt.legend()
#plt.show()
#T2T.plot(kind='bar', label='top t2')
#plt.legend()
#plt.show()
#T2D.plot(kind='bar', label='down t2')
#plt.legend()
#plt.show()
#RunT.plot(kind='bar', label='top run')
#plt.legend()
#plt.show()
#RunD.plot(kind='bar', label='down run')
#plt.legend()
#plt.show()
#
#
#data.plot(kind='scatter', x=['Swim time'], y=['Cycle time'])
#a = data[['Swim time']]
#b = data['Cycle time']
#model = LinearRegression()
#model.fit(a, b)
#b_pred = model.predict(a)
#data['b_pred'] = b_pred
#r2 = r2_score(b, b_pred)
#plt.plot(data['Swim time'], data['b_pred'], color='black', label=f'Regression line (R² = {r2:.2f})')
#plt.title('Swim vs cycle time', fontsize=14)
#plt.legend()
#plt.show()
#
#data.plot(kind='scatter', x=['Run time'], y=['Cycle time'])
#a = data[['Run time']]
#b = data['Cycle time']
#model = LinearRegression()
#model.fit(a, b)
#b_pred = model.predict(a)
#data['b_pred'] = b_pred
#r2 = r2_score(b, b_pred)
#plt.plot(data['Run time'], data['b_pred'], color='black', label=f'Regression line (R² = {r2:.2f})')
#plt.title('Run vs cycle time', fontsize=14)
#plt.legend()
#plt.show()
#
#data.plot(kind='scatter', x=['Rank swim'], y=['Rank cycle'])
#a = data[['Rank swim']]
#b = data['Rank cycle']
#model = LinearRegression()
#model.fit(a, b)
#b_pred = model.predict(a)
#data['b_pred'] = b_pred
#r2 = r2_score(b, b_pred)
#plt.plot(data['Rank swim'], data['b_pred'], color='black', label=f'Regression line (R² = {r2:.2f})')
#plt.title('Swim vs cycle rank', fontsize=14)
#plt.legend()
#plt.show()
#
#data.plot(kind='scatter', x=['Time T1'], y=['Time T2'])
#a = data[['Time T1']]
#b = data['Time T2']
#model = LinearRegression()
#model.fit(a, b)
#b_pred = model.predict(a)
#data['b_pred'] = b_pred
#r2 = r2_score(b, b_pred)
#plt.plot(data['Time T1'], data['b_pred'], color='black', label=f'Regression line (R² = {r2:.2f})')
#plt.title('Swim vs cycle time', fontsize=14)
#plt.legend()
#plt.show()









