import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

colors = {
    'LACNN': 'tab:blue',
    'CNN': 'tab:orange',
    'NN': 'tab:green',
    'LLR': 'tab:red',
    'LANN': 'tab:purple',
    'Naive': 'tab:cyan',
}

def fancy(csv_data_path, save_to):
    df = pd.read_csv(csv_data_path)
    keys = df.keys()
    plt.figure(figsize=(10, 7.5))

    for i in range(len(keys)-1):
        # r, g, b = tableau20[i]
        # tableau20[i] = (r/255, g/255, b/255)
        x = df['epoch']
        y = df[str(keys[i+1])]
        xnew = np.linspace(x.min(),x.max(),300)
        spl = make_interp_spline(x, y, k=2)
        power_smooth = spl(xnew)
        # plt.plot(df['epoch'], df[str(keys[i+1])], label=keys[i+1], lw=1.5)
        plt.plot(
            x,
            y,
            '--',
            label=keys[i+1],
            lw=1.5,
            color=colors[str(keys[i+1])]
        )
        # plt.text(-1, df[str(keys[i+1])][0], str(keys[i+1]), fontsize=12, color=tableau20[i]) 
    plt.plot(
            x,
            [0.16 for _ in range(len(x))],
            '--',
            label='Naive',
            lw=1.5,
            color=colors['Naive']
        )
    
    plt.legend()
    plt.xlim([0, max(df['epoch'])])
    plt.ylim([0, 0.7])
    plt.xlabel('epoch')
    plt.ylabel('negative log-likelihood loss')
    # plt.title('BCEwithLogitsLoss')
    # plt.grid(True)
    # plt.show()
    plt.savefig(save_to, bbox_inches='tight')
