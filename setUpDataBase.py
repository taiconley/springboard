from dbPostgres import DB
import pandas as pd
from os import listdir
from os.path import isfile, join
import credentials

dbName = 'nba'  # 'pi.db'
userName = credentials.userName
userPass = credentials.userPass
db = DB(userName=userName, userPass=userPass, dataBaseName=dbName)


# fieldList = '''(
# URL TEXT,
# GameType TEXT,
# Location TEXT,
# Date TEXT,
# Time TEXT,
# WinningTeam TEXT,
# Quarter TEXT,
# SecLeft TEXT,
# AwayTeam TEXT,
# AwayPlay TEXT,
# AwayScore TEXT,
# HomeTeam TEXT,
# HomePlay TEXT,
# HomeScore TEXT,
# Shooter TEXT,
# ShotType TEXT,
# ShotOutcome TEXT,
# ShotDist TEXT,
# Assister TEXT,
# Blocker TEXT,
# FoulType TEXT,
# Fouler TEXT,
# Fouled TEXT,
# Rebounder TEXT,
# ReboundType TEXT,
# ViolationPlayer TEXT,
# ViolationType TEXT,
# TimeoutTeam TEXT,
# FreeThrowShooter TEXT,
# FreeThrowOutcome TEXT,
# FreeThrowNum TEXT,
# EnterGame TEXT,
# LeaveGame TEXT,
# TurnoverPlayer TEXT,
# TurnoverType TEXT,
# TurnoverCause TEXT,
# TurnoverCauser TEXT,
# JumpballAwayPlayer TEXT,
# JumpballHomePlayer TEXT,
# JumpballPoss TEXT
# )'''

# db_table = "pbp"

# db.buildTable(tableName=db_table, fieldList=fieldList)

# print(db.tables)


mypath = './data'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]


def process(file_name):
    file_name = 'data/{}'.format(file_name)
    df = pd.read_csv(file_name)
    print(df.head())
    db.DFtoDB(df, 'pbp')


for file in files:
    process(file)
