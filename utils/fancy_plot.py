import matplotlib.pyplot as plt  
import pandas as pd

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def fancy(csv_data_path):
    df = pd.read_csv(csv_data_path)
    keys = df.keys()
    plt.figure(figsize=(10, 7.5))
    plt.xlim(1, df['epoch'].max())

    for i in range(len(keys)-1):
        r, g, b = tableau20[i]
        tableau20[i] = (r/255, g/255, b/255)
        plt.plot(df['epoch'], df[str(keys[i+1])], label=keys[i+1], lw=1.5, color=tableau20[i])
        # plt.text(-1, df[str(keys[i+1])][0], str(keys[i+1]), fontsize=12, color=tableau20[i]) 
    
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train and validation negative log likelihood values after 20 epochs')
    plt.grid(True)
    plt.show()
