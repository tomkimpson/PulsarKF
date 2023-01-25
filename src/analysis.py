



id = "51f7e999"
results_file = f"../results/{id}/{id}_result.json"



import json
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

def give_me_some_numbers(fname):

    f = open(fname)
    d = json.load(f)
    

    
    log_evidence =  d["log_evidence"]

    posterior = pd.DataFrame(d["posterior"]["content"])

    
    sns.histplot(data=posterior, x="omega",log_scale=True,kde=True)
    plt.show()











give_me_some_numbers(results_file)