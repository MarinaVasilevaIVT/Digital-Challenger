import pandas as pd
import numpy as np
import collections

data = pd.read_csv("data.csv", encoding = 'UTF-8', delimiter=',')
data.head()
#print(data)

TN = pd.read_csv('TeamName.csv')

teamList = TN['Team Name'].tolist()
#print(teamList)

deleteTeam = [x for x in pd.unique(data['Команда']) if x not in teamList]
#print(pd.unique(data['Команда']))

for name in deleteTeam:
    data = data[data['Команда'] != name]
    data = data[data['Противник'] != name]
data = data.reset_index(drop=True)

#print(len(data))

def GetSeasonTeamStat(team, year):
    WonSet = 0 # Количество выигранных сетов
    LostSet = 0 # Количество проигранных сетов

    gameWin = 0 #Выиграно
    gameLost = 0 #Проиграно

    totalScore = 0 #Количество набранных очков

    matches = 0 #Количество сыгранных матчей
    
    serve = 0 #Количество подач
    MistakesServe = 0 #Количество ошибок при подачах
    ace = 0 #Эйс
    
    attack = 0 #Очки с атак
    MistakesAttack = 0 #Ошибки при атаках
    
    Receiving = 0 #Количество приемов
    MistakesReceiving = 0 #Ошибки при приемах
    
    Block = 0 #Количество блоков
  
    for i, row in data.iterrows():
        if (((row['Год'] == year) and (row['Команда'] == team))):
            matches += 1
            WonSet += data['Количество выигранных сетов'][i]
            LostSet += data['Количество проигрышных сетов'][i]

            if (data['Количество выигранных сетов'][i] == 3 and (data['Количество проигрышных сетов'][i] == 0 or data['Количество проигрышных сетов'][i] == 1)):
                totalScore += 3
                gameWin += 1
            
            elif (data['Количество выигранных сетов'][i] == 3 and data['Количество проигрышных сетов'][i] == 2):
                totalScore += 2
                gameWin += 1
            
            elif (data['Количество выигранных сетов'][i] == 2 and data['Количество проигрышных сетов'][i] == 3):
                totalScore += 1
                gameLost +=1
            
            elif ((data['Количество выигранных сетов'][i] == 1 or data['Количество выигранных сетов'][i] == 0) and data['Количество проигрышных сетов'][i] == 3):
                gameLost +=1
            
            serve += data['Общее количество подач'][i]
            MistakesServe += data['Ошибки при подаче'][i]
            ace += data['ЭЙС'][i]
            
            attack += data['Общее количество атак'][i]
            MistakesAttack += data['Ошибки на атаках'][i]
            
            Receiving += data['Общее количество приемов'][i]
            MistakesReceiving += data['Ошибки при приемах'][i]
            
            Block += data['Очки на блок'][i]


    return [gameWin, gameLost, 
            WonSet, LostSet, totalScore,
            serve, MistakesServe, ace, 
            attack, MistakesAttack,
            Receiving, MistakesReceiving,
            Block]

returnNames = ["Выиграно игр", "Проиграно игр",
               "\nКоличество выигранных сетов", "Количество проигранных сетов", "\nНабрано очков",
               "\nОбщее количество подач", "Количество ошибок при подачах", "ЭЙС",
               "\nОчки с атак", "Ошибки при атаках", 
               "\nКоличество приемов", "Ошибки при приемах",
               "\nКоличество заблокированных мячей"]

for i, n in zip(returnNames, GetSeasonTeamStat('ЗЕНИТ', 2023)):
        print(i, n)

def GetSeasonAllTeamStat(season):
    annual = collections.defaultdict(list)
    for team in teamList:
        team_vector = GetSeasonTeamStat(team, season)
        annual[team] = team_vector
    return annual

#print(GetSeasonAllTeamStat(2021))

def GetTrainingData(seasons):
    totalNumGames = 0
    for season in seasons:
        annual = data[data['Год'] == season]
        totalNumGames += len(annual.index)
    numFeatures = len(GetSeasonTeamStat('ЗЕНИТ', 2023)) #случайная команда для определения размерности
    xTrain = np.zeros((totalNumGames, numFeatures))
    yTrain = np.zeros(totalNumGames)
    indexCounter = 0
    for season in seasons:
        team_vectors = GetSeasonAllTeamStat(season)
        annual = data[data['Год'] == season]
        numGamesInYear = len(annual.index)
        xTrainAnnual = np.zeros((numGamesInYear, numFeatures))
        yTrainAnnual = np.zeros((numGamesInYear))
        counter = 0
        for index, row in annual.iterrows():
            team = row['Команда']
            t_vector = team_vectors[team]
            rivals = row['Противник']
            r_vector = team_vectors[rivals]
           
            diff = [a - b for a, b in zip(t_vector, r_vector)]
            
            if len(diff) != 0:
                xTrainAnnual[counter] = diff
            if team == row['Победитель']:
                yTrainAnnual[counter] = 1
            else: 
                yTrainAnnual[counter] = 0
            counter += 1   
        xTrain[indexCounter:numGamesInYear+indexCounter] = xTrainAnnual
        yTrain[indexCounter:numGamesInYear+indexCounter] = yTrainAnnual
        indexCounter += numGamesInYear
    return xTrain, yTrain


years = range(2021,2024)
xTrain, yTrain = GetTrainingData(years)
#print(xTrain, yTrain)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xTrain, yTrain)

def createGamePrediction(team1_vector, team2_vector):
    diff = [[a - b for a, b in zip(team1_vector, team2_vector)]]
    predictions = model.predict(diff)
    return predictions

team1_name = "ЗЕНИТ"
team2_name = "АСК"

team1_vector = GetSeasonTeamStat(team1_name, 2024)
team2_vector = GetSeasonTeamStat(team2_name, 2024)

print ('Вероятность, что выиграет ' + team1_name + ':', createGamePrediction(team1_vector, team2_vector))
print ('Вероятность, что выиграет ' + team2_name + ':', createGamePrediction(team2_vector, team1_vector))

for team_name in teamList:
    team1_name = "ЗЕНИТ"
    team2_name = team_name
    
    if(team1_name != team2_name):
        team1_vector = GetSeasonTeamStat(team1_name, 2024)
        team2_vector = GetSeasonTeamStat(team2_name, 2024)

        print(team1_name, createGamePrediction(team1_vector, team2_vector), " - ", team2_name, createGamePrediction(team2_vector, team1_vector,))