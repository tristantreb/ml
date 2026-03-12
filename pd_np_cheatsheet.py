import pandas as pd

# df = employee.drop_duplicates(subset='salary')
# df.sort_values(by='salary', ascending=False)

def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    df = employee.drop_duplicates(subset='salary')
    nunique = df.salary.nunique()
    if nunique < N or N < 1:
        topn = None
    else:
        df = df.sort_values(by='salary', ascending=False)
        topn = df.salary.values[N-1]
    topdf = pd.DataFrame([topn], columns=[f'getNthHighestSalary({N})'])
    return topdf

    # range(start, stop, step)
# np.where(cond, x, y)

# ranks = df.score.rank(ascending=False, method='dense').values
# df = df.drop_duplicates(subset='salary')


# Compute consecutive numbers in logs.num
def consecutive_numbers(logs: pd.DataFrame) -> pd.DataFrame:
    logs['stds'] = logs.num.rolling(3, min_periods=3).std()
    triplets = logs.loc[logs.stds == 0, 'num']
    ans = triplets.unique()
    return pd.DataFrame(ans, columns=['ConsecutiveNums'])

# Use diff
def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.sort_values(by=["recordDate"]).reset_index(drop=True)
    day_mask = weather.recordDate.diff().apply(lambda x: x.days) == 1
    temp_mask = weather.temperature.diff().apply(lambda x: x>0)
    ans = weather[day_mask & temp_mask].id.values
    return pd.DataFrame(ans, columns=["Id"])


# DATETIME
from datetime import datetime

dt = datetime.now()

dt.strftime("%Y-%m-%d")          # '2026-03-03'
dt.strftime("%Y-%m-%d %H:%M:%S") # '2026-03-03 14:30:45'
dt.strftime("%d/%m/%Y")          # '03/03/2026'

# Code	Meaning	Example
# %Y	4-digit year	2026
# %m	Month (zero-padded)	03
# %d	Day (zero-padded)	03
# %H	Hour (24h)	14
# %M	Minute	30
# %S	Second	45
# %B	Full month name	March
# %A	Full weekday name	Tuesday
