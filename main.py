import pandas as pd
from pandasgui import show

dfStudents = pd.read_csv('./StudentPerformanceFactors.csv')

show(dfStudents)