import os

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model
from utils.layer_utils import AttentionLSTM


#Build LSTM-FCN model
def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):


    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))


    #Set up LSTM
    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)

    #First Convolutional Layer
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    #Second Convolutional Layer
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    #Third Convolutional Layer
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    return model

#This is the LSTM-FCN with attention mechanism. This is not used in the current model
def generate_alstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == "__main__":

    dataset_map = [('RSS', 0)]
    print("Num datasets : ", len(dataset_map))
    print()

    base_log_name = '%s_%d_cells_new_datasets.csv'
    base_weights_dir = '%s_%d_cells_weights/'

    MODELS = [
        ('lstmfcn', generate_lstmfcn),
  #      ('alstmfcn', generate_alstmfcn),
    ]

    # Number of cells
    CELLS = [8, 64, 128]

    # Normalization scheme
    # Normalize = False means no normalization will be done
    # Normalize = True / 1 means sample wise z-normalization
    # Normalize = 2 means dataset normalization.
    normalize_dataset = True

    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        for cell in CELLS:
            successes = []
            failures = []

            if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
                file = open(base_log_name % (MODEL_NAME, cell), 'w')
                file.write('%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy'))
                file.close()

            for dname, did in dataset_map:

                MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[did]
                NB_CLASS = NB_CLASSES_LIST[did]

                # release GPU Memory
                K.clear_session()

                file = open(base_log_name % (MODEL_NAME, cell), 'a+')

                weights_dir = base_weights_dir % (MODEL_NAME, cell)

                

                if not os.path.exists('weights4/' + weights_dir):
                    os.makedirs('weights4/' + weights_dir)

                dataset_name_ = weights_dir + dname

                # try:
                model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)

                print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

                # comment out the training code to only evaluate !
                train_model(model, did, dataset_name_, epochs=50, batch_size=2,
                            normalize_timeseries=normalize_dataset)

                acc = evaluate_model(model, did, dataset_name_, batch_size=2,
                                     normalize_timeseries=normalize_dataset)

                s = "%d,%s,%s,%0.6f\n" % (did, dname, dataset_name_, acc)

                file.write(s)
                file.flush()

                successes.append(s)

                # except Exception as e:
                #     traceback.print_exc()
                #
                #     s = "%d,%s,%s,%s\n" % (did, dname, dataset_name_, 0.0)
                #     failures.append(s)
                #
                #     print()

                file.close()

            print('\n\n')
            print('*' * 20, "Successes", '*' * 20)
            print()

            for line in successes:
                print(line)

            print('\n\n')
            print('*' * 20, "Failures", '*' * 20)
            print()

            for line in failures:
                print(line)
