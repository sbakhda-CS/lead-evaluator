'''
NOT READY AT ALL -- just copied and pasted functions from previous scripts I've used for pre-processing
'''


import pandas as pd
import re


# remove special characters and words that are not in the model's vocabulary
def clean_input(raw_bio):
    # if no bio, return a no bio indicator "NONE"
    if type(raw_bio) == float:
        return "NONE"

    # remove special characters
    bio = re.sub(r'[^\w]', ' ', raw_bio.lower() ).split()

    i=0
    # check if each word in bio is in model's vocab
    while i < len(bio):
        cur_word = bio[i]
        # if word is not in model's vocab, remove word from input text
        if cur_word not in model.vocab:
            bio.remove(cur_word)
            i-=1
        i+=1

    # input now cleaned and ready to go into model
    cleaned_input = bio

    return cleaned_input


# main function for preprocessing CSV file
def preprocess(file, normal_cols, one_hot_cols, label_cols):
    df = pd.read_csv(file)

    label_col = make_label_col(label_cols, df)
    numerical_cols = make_normal_cols(normal_cols, df)
    categorical_cols = make_one_hot_cols(one_hot_cols, df)


# constructs label column from label columns
def make_label_col(label_cols, df):
    label_col = []
    for i in range(0, len(df[label_cols[0]])):
        if any(df.col_name[i] > 0 for col_name in label_cols):
            label_col.append(1)
        else:
            label_col.append(0)
    return label_col


# constructs numerical cols
def make_normal_cols(normal_cols, df):
    normal_cols_ret = []
    for col_name in normal_cols:
        normal_cols_ret.append(df.col_name)



# return : [list of lists] of categorical data transformed into one-hot columns
def one_hot(data):
    values = np.array(data)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded



if __name__ == "__main__":
    file_name = 'hubspot.csv'
    preprocess(file=file_name, all_cols=[], one_hot_cols=[], label_cols=[])