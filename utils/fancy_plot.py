import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

colors = {
    'SACNN': 'tab:blue',
    'CNN': 'tab:orange',
    'NN': 'tab:green',
    'LLR': 'tab:red',
    'SANN': 'tab:purple',
    'SALLR': 'tab:cyan'
}

def fancy(csv_data_path, save_to):
    df = pd.read_csv(csv_data_path)
    keys = df.keys()
    plt.figure(figsize=(10, 7.5))
    plt.xlim(1, df['epoch'].max())

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
            xnew,
            power_smooth,
            label=keys[i+1],
            lw=1.5,
            color=colors[str(keys[i+1])]
        )
        # plt.text(-1, df[str(keys[i+1])][0], str(keys[i+1]), fontsize=12, color=tableau20[i]) 
    
    plt.legend()
    plt.xlim([0, max(df['epoch'])])
    plt.ylim([0, 0.4])
    plt.xlabel('Epoch')
    plt.ylabel('NLL Value')
    plt.title('BCEwithLogitsLoss')
    plt.grid(True)
    # plt.show()
    plt.savefig(save_to, bbox_inches='tight')
