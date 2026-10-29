#UTILS - Functions used by multiple files and main
from datetime import datetime

#Datetime
def getTime():
    return datetime.now().strftime("%H:%M:%S")