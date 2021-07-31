import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''Action labels are provided as:
0 - vertical hand
1 - Wave Up and Down
2 - Push and Pull 

Actions are completed in 10 second time intervals. Since beacon frames are used, the frame intervals are not constant. 
Beacon frames range  between 27-32 frames per 10 seconds. 
Therefore 27 data points were taken for each gesture. Other data points were discarded.

Currently 20 samples generated for each action per session. Samples are taken consecutively one after the other. To limit bias
the samples are then shuffled and 15 samples are used as training samples and 5 samples are used as test samples

Default code currently converts 240 sized dataset into a training and test file for model training and testing

Uncomment the followng lines below to get a 4 split for 4 fold validation


'''

#CSV files are taken from data extracted from wireshark
csv_files = ["CSV_files/vertical_hand_1.csv", "CSV_files/wave_up_down_1.csv", "CSV_files/push_pull_1.csv", 
             "CSV_files/vertical_hand_2.csv", "CSV_files/wave_up_down_2.csv", "CSV_files/push_pull_2.csv",
             "CSV_files/vertical_hand_3.csv", "CSV_files/wave_up_down_3.csv", "CSV_files/push_pull_3.csv",
             "CSV_files/vertical_hand_4.csv", "CSV_files/wave_up_down_4.csv", "CSV_files/push_pull_4.csv"]
action_ids = [1,2,3,
              1,2,3,
              1,2,3,
              1,2,3]

train_list = []
test_list = []
split1_list = []
split2_list = []
split3_list = []
split4_list = []


for action, csv in zip(action_ids,csv_files):
    df = pd.read_csv(csv).dropna()


    #Remove any RSS values with zero dBm
    df3 = df[df["Signal strength (dBm)"] != "0 dBm"] 
    RSS_signals = df3["Signal strength (dBm)"].str.replace('\sdBm', '').astype(int)
    time = df3["Time"]
    new_df = pd.concat([time, RSS_signals], axis=1)

    #Convert all times to the closest 10 second floor
    new_df['secs'] = (new_df['Time'] / 10).apply(np.floor).astype(int) *10
    #Group rows into 10 second intervals
    l = new_df.groupby('secs')['Signal strength (dBm)']

    full_list = []
    for idx, (name, group) in enumerate(l):

        #Remove any data points above 200 seconds 
        if (name >= 200):
            break

        if (len(group) < 27):
            continue

        sec_list = []
        for n in group:
            sec_list.append(n)

            #Collect only the first 27 data points in the interval
            if(len(sec_list) == 27):
                break

        full_list.append(sec_list)

    #Create dataframe and shuffle samples
    action_df = pd.DataFrame((full_list)).sample(frac = 1)

    #Add action labels to gesture samples
    action_df.insert(0, "action", action)

    #Split samples into training and test samples. 15 samples to training and 5 samples to testing
    (train,test) = np.split(action_df, [15])

    #Uncomment this code to generate 4 splits
    #(split1,split2,split3,split4) = np.split(action_df, 4)

    train_list.append(train)
    test_list.append(test)


    #Uncomment this code to generate 4 splits
    # split1_list.append(split1)
    # split2_list.append(split2)
    # split3_list.append(split3)
    # split4_list.append(split4)



train_arr = np.concatenate(train_list)
test_arr = np.concatenate(test_list)

np.random.shuffle(train_arr)
np.random.shuffle(test_arr)

#Generate training and test file
pd.DataFrame(train_arr).to_csv("data/RSS_TRAIN_240", header=False, index=False)
pd.DataFrame(test_arr).to_csv("data/RSS_TEST_240", header=False, index=False)


# Uncomment the code below to generate 4 fold splits 


# split1_arr = np.concatenate(split1_list)
# split2_arr = np.concatenate(split2_list)
# split3_arr = np.concatenate(split3_list)
# split4_arr = np.concatenate(split4_list)


# np.random.shuffle(split1_arr)
# np.random.shuffle(split2_arr)
# np.random.shuffle(split3_arr)
# np.random.shuffle(split4_arr)


# split1_train = np.concatenate((split2_arr, split3_arr,split4_arr))
# split1_test = split1_arr


# pd.DataFrame(split1_train).to_csv("RSS_TRAIN1", header=False, index=False)
# pd.DataFrame(split1_test).to_csv("RSS_TEST1", header=False, index=False)

# split2_train = np.concatenate((split1_arr, split3_arr,split4_arr))
# split2_test = split2_arr


# pd.DataFrame(split2_train).to_csv("RSS_TRAIN2", header=False, index=False)
# pd.DataFrame(split2_test).to_csv("RSS_TEST2", header=False, index=False)

# split3_train = np.concatenate((split2_arr, split1_arr,split4_arr))
# split3_test = split3_arr


# pd.DataFrame(split3_train).to_csv("RSS_TRAIN3", header=False, index=False)
# pd.DataFrame(split3_test).to_csv("RSS_TEST3", header=False, index=False)

# split4_train = np.concatenate((split2_arr, split3_arr,split1_arr))
# split4_test = split4_arr


# pd.DataFrame(split4_train).to_csv("RSS_TRAIN4", header=False, index=False)
# pd.DataFrame(split4_test).to_csv("RSS_TEST4", header=False, index=False)
        

