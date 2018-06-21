from gensim.models import KeyedVectors
from scipy import spatial
import numpy as np
import re
import pandas as pd

# names of sheets and columns in xlsx file for Company ID and Bio
COMPANY_ID_SHEET = 'Contact Details'
COMPANY_ID_COL = 'Company ID'
BIO_SHEET = 'Bio'
BIO_COL = 'LinkedIn Bio'

# number of dimensions in pre-trained vectors
NUM_VECTOR_DIMENSIONS = 50

# file with pre-trained vectors (GloVe 50d or 100d vectors in this case)
VECTORS_FILE_NAME = 'glove.6B.50d.txt'

# file with Bio data
DATA_FILE_NAME = 'companies.xlsx'

# load pre-trained model
model = KeyedVectors.load_word2vec_format(VECTORS_FILE_NAME, binary=False)

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

# get one averaged vector representation for a given document
def get_averaged_bio_vector(bio):

    # if company has no bio, return array of zeros
    if bio == "NONE":
        return np.zeros(shape=NUM_VECTOR_DIMENSIONS)

    # model's representation of each word in bio: each word is a 50d pre-trained vector
    model_representation = model[bio]

    # averages each word vector to create one document vector for the whole bio
    averaged_vector = np.mean(model_representation, axis=0)

    return averaged_vector

# get cosine similarity of two document vectors (float from 0 - 1, 1=identical, 0=completely unrelated)
def cosine_similarity(bio1, bio2):
    # get averaged vectors for both bios
    bio_vector_1 = get_averaged_bio_vector(bio1)
    bio_vector_2 = get_averaged_bio_vector(bio2)

    # cosine similarity measurement for each bio vector
    cosine_similarity = 1 - spatial.distance.cosine(bio_vector_1, bio_vector_2)

    return cosine_similarity

def main():
    ret = []
    # get sheet & list for Company ID
    company_id_sheet = pd.read_excel(DATA_FILE_NAME, sheet_name=COMPANY_ID_SHEET)
    company_id_list = company_id_sheet[COMPANY_ID_COL].values.tolist()[1:]

    # get sheet & list for Bio
    bio_sheet = pd.read_excel(DATA_FILE_NAME, sheet_name=BIO_SHEET)
    bio_list = bio_sheet[BIO_COL].values.tolist()[1:]

    num_bios = len(bio_list)
    i=0
    # append dicts for each company to return array
    while i < num_bios:
        # create dict with {Company ID : averaged bio vector}
        dict = {company_id_list[i] : get_averaged_bio_vector(clean_input(bio_list[i]))}

        # add dict to return array
        ret.append(dict)
        i+=1

    return ret

main()