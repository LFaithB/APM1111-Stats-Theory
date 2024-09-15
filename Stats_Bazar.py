#!/usr/bin/env python
# coding: utf-8

# In[3]:


import polars as pl
import statistics
from scipy import stats as st
from math import sqrt
from IPython.core.display import HTML, display

data = [88, 45, 53, 86, 33, 86, 85, 30, 89, 53, 41, 96, 56, 38, 62, 
        71, 51, 86, 68, 29, 28, 47, 33, 37, 25, 36, 33, 94, 73, 46, 
        42, 34, 79, 72, 88, 99, 82, 62, 57, 42, 28, 55, 67, 62, 60,
        96, 61, 57, 75, 93, 34, 75, 53, 32, 28, 73, 51, 69, 91, 35]

valc = 0
for num in data:
    if str(num).isdigit():
        valc += 1 

value = []  
stats = []


stats.append("Valid Count")
value.append(f"{val_count}")

stats.append("Mode a")
value.append(f"{min(statistics.multimode(data)):.3f}")

stats.append("Median")
value.append(f"{statistics.median(data):.3f}")

stats.append("Mean")
value.append(f"{statistics.mean(data):.3f}")

stats.append("Std. Deviation")
value.append(f"{statistics.stdev(data):.3f}")

stats.append("Variance")
value.append(f"{statistics.variance(data):.3f}")

stats.append("Skewness")
value.append(f"{st.skew(data):.3f}")

stats.append("Std. Error of Skewness")
value.append(f"{sqrt(6 / len(data)):.3f}")

stats.append("Kurtosis")
value.append(f"{st.kurtosis(data):.3f}")

stats.append("Std Error of Kurtosis")
value.append(f"{sqrt(24 / len(data)):.3f}")

stats.append("Minimum")
value.append(f"{min(data):.3f}")

stats.append("Maximum")
value.append(f"{max(data):.3f}")

stats.append("25th Percentile")
value.append(f"{st.scoreatpercentile(data, 25):.3f}")

stats.append("50th Percentile")
value.append(f"{st.scoreatpercentile(data, 50):.3f}")

stats.append("75th Percentile")
value.append(f"{st.scoreatpercentile(data, 75):.3f}")

stats.append("90th Percentile")
value.append(f"{st.scoreatpercentile(data, 90):.3f}")

stats.append("95th Percentile")
value.append(f"{st.scoreatpercentile(data, 95):.3f}")

table_input = {"Statistic": stats, "Score": val}
df = pl.DataFrame(table_input)

def display_html(df):
    html_content = df.to_pandas().to_html(index=False, border=0)
    return HTML(html_content)

html_table = display_html(df)
note_html = "<p><strong><span style='font-size: smaller;'>a</span></strong> More than one mode exists, only the first is reported.</p>"
full_html = f"{html_table.data}{note_html}"

display(HTML(full_html))

