import requests


url="http://yonm10.pythonanywhere.com/predict"

player={
 'age': 31,  
 'overall': 94,  # range from 0 to 99
 'value': '€110.5M', # Market value of player
 'wage': '€565K',
 'special': 2202, # it looks like some sum of stat 
 'international_reputation': 5.0, # range from 0 to 5 
 'weak_foot': 4.0, # range from 0 to 5 
 'skill_moves': 4.0, # range from 0 to 5 
 'work_rate': 'Medium/ Medium', # attacking work rate/ defending work rate .Possible values are Low, Medium ,High
 'position': 'RF', # Position of player
 'ls': '88+2', # rate of player play in position ls , range from 0 to 99
 'st': '88+2',
 'rs': '88+2',
 'lw': '92+2',
 'lf': '93+2',
 'cf': '93+2',
 'rf': '93+2',
 'rw': '92+2',
 'lam': '93+2',
 'cam': '93+2',
 'ram': '93+2',
 'lm': '91+2',
 'lcm': '84+2',
 'cm': '84+2',
 'rcm': '84+2',
 'rm': '91+2',
 'lwb': '64+2',
 'ldm': '61+2',
 'cdm': '61+2',
 'rdm': '61+2',
 'rwb': '64+2',
 'lb': '59+2',
 'lcb': '47+2',
 'cb': '47+2',
 'rcb': '47+2',
 'rb': '59+2', # rating of player play in position rb , range from 0 to 99
 'crossing': 84.0, # this line and below(except release clause) attribute is about technical thing , range from 0 to 99
 'finishing': 95.0,
 'headingaccuracy': 70.0,
 'shortpassing': 90.0,
 'volleys': 86.0,
 'dribbling': 97.0,
 'curve': 93.0,
 'fkaccuracy': 94.0,
 'longpassing': 87.0,
 'ballcontrol': 96.0,
 'acceleration': 91.0,
 'sprintspeed': 86.0,
 'agility': 91.0,
 'reactions': 95.0,
 'balance': 95.0,
 'shotpower': 85.0,
 'jumping': 68.0,
 'stamina': 72.0,
 'strength': 59.0,
 'longshots': 94.0,
 'aggression': 48.0,
 'interceptions': 22.0,
 'positioning': 94.0,
 'vision': 94.0,
 'penalties': 75.0,
 'composure': 96.0,
 'marking': 33.0,
 'standingtackle': 28.0,
 'slidingtackle': 26.0,
 'gkdiving': 6.0,
 'gkhandling': 11.0,
 'gkkicking': 15.0,
 'gkpositioning': 14.0,
 'gkreflexes': 8.0,
 'release_clause': '€226.5M'} 

response=requests.post(url,json=player).json()
print(response)