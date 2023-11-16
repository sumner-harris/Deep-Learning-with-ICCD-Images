import pandas as pd
import numpy as np

def load_df(datafile, normalize_PTE1E2 = False):
    #read the json data file
    df = pd.read_json(datafile)

    ICCD = []
    T = []
    E1 = []
    E2 = []
    P = []
    s0 = []
    s1 = []
    J = []
    
    #iterate through the df and
    for i in range(len(df)):
        ICCD.append(np.array(df['ICCD'][i]).swapaxes(2,0).swapaxes(1,2).reshape(1,50,40,40))
        E1.append(df['params'][i][2])
        E2.append(df['params'][i][3])
        T.append(df['params'][i][1])
        P.append(df['params'][i][0])
        s0.append(df['LR_fits'][i][0])
        s1.append(df['LR_fits'][i][1])
        J.append(df['LR_fits'][i][2])
        
    if normalize_PTE1E2:
        datadict = {'ICCD':ICCD, 'P':np.array(P)/np.array(P).max(), 'T':np.array(T)/np.array(T).max(),
                    'E1':np.array(E1)/np.array(E1).max(), 'E2':np.array(E2)/np.array(E2).max(),
                    's0':np.array(s0)*10, 's1':np.array(s1), 'J':np.array(J)*1000
                    }
        
    else:
        datadict = {'ICCD':ICCD, 'P':np.array(P), 'T':np.array(T),
            'E1':np.array(E1), 'E2':np.array(E2),
            's0':np.array(s0)*10, 's1':np.array(s1), 'J':np.array(J)*1000
            }

    df = pd.DataFrame(datadict)
    
    return df