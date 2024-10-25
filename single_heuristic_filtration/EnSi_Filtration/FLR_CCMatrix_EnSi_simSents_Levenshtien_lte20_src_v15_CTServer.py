#v3 - levenstien distance categories for 3 segements
#v5 - Levenshtein distance calculation for n-1 batch 17.06.24
#v6 - select dataframe accoridng to sliding window and calculate the Levenshtien distance.
#v7 - Addressing warnings-24.08.2024
# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   subset_current_tgt_df["current_tgt_sent"] = getattr(row, "tgt_sents")
# FLR_CCMatrix_SiTa_simSents_Levenshtien_lte5_tgt_v6.py:149: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
#Chained Index = https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#copy-on-write-chained-assignment
#v11-06.07.2024
#Lev_distance calculation -from Levenshtein import distance
#write the (1) ref sentences & (2) overlapping sentences into two files
#v14
#paralllize the calculation with mapply.
#v15
#removing sentences considering overlapping % ie. less than 60%
#
#---------------------------------------------------------------------------------------------------------------------------------------------

#imports
import time
import os
import string
import pandas as pd
import mapply
from Levenshtein import distance


#debug
debug = False
startTime=time.time()

#parameters
resume_experiment = False
levenshtein_distance_threshold = 20
sliding_window_size = 10
min_ngram_overlap = 4

#inputs
data_dir="/home/uomadm/p4_parallel_data_curation/data/CCMatrix/en-si.txt"
input_file = "CCMatrix-scores.en-si.csv"
df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")
print("Original Dataset size : {}".format(len(df)))
print(df) if debug else None


#outputs
corpus = "ccmatrix"
sub_dir = "csv_files"
exp_file_name="sentSim_Levenshtien_lte{}".format(levenshtein_distance_threshold)
src_lang="en"
tgt_lang="si"
output_dir="/home/uomadm/p4_parallel_data_curation/data/{}_{}/".format(corpus, exp_file_name)

if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
else:
    print("Directory exists...")

#output files
laser3_src_sorted_csv_file="{}-{}-laser3-sorted-src-{}.train.csv".format(src_lang, tgt_lang, exp_file_name)
xlmr_src_sorted_csv_file="{}-{}-xlmr-sorted-src-{}.train.csv".format(src_lang, tgt_lang, exp_file_name)
labse_src_sorted_csv_file="{}-{}-labse-sorted-src-{}.train.csv".format(src_lang, tgt_lang, exp_file_name)

#tgt_ref & tgt_similar sentences
ref_src_sents_csv_file = "{}-{}-ref_src_sents-Lev{}.train.csv".format(src_lang, tgt_lang, levenshtein_distance_threshold)
src_similar_sents_csv_file = "{}-{}-similar_src_sents-Lev{}.train.csv".format(src_lang, tgt_lang, levenshtein_distance_threshold)

#################################################################
#Add columns
###############################################################

#remove punctuations from string - creates a translation table that maps punctuations to None
translator = str.maketrans('', '', string.punctuation +'–“’…')

#remove punctuations & special characters
def get_punct_rm_sentence(text):
    return str(text).translate(translator)

#get src_length
def get_src_sent_length(text):
    return len(text.split())

#returns ngram chunnks
def get_substring_ngram_segments(text):
    ngram_segments = []
    sentTkns = [str(token) for token in str(text).strip().split()]
    if len(sentTkns) >= 0:
        ngram_segments = [" ".join(sentTkns[i:i+min_ngram_overlap]) for i in range(len(sentTkns) - min_ngram_overlap+1)]
    return ngram_segments

#get values
df["src_mod_sents"] = df["src_sents"].apply(get_punct_rm_sentence)
df["src_sent_words"] = df["src_mod_sents"].apply(lambda x : str(x).split())
df["src_sent_chars"] = df["src_mod_sents"].apply(lambda x : list(str(x).replace(" ", "")))
df["src_sent_chars_length"] = df["src_sent_chars"].apply(lambda x : len(x))
df["src_substring_ngram_segments"] = df["src_mod_sents"].apply(get_substring_ngram_segments)

###################################################################
#deduplication
###################################################################
#drop duplicates dataframe
print("\nDeduplicating:")
print("----------------")
src_dedup_df = df.drop_duplicates("src_mod_sents", keep="first").iloc[:,:]
print("Size of the de-duplicated/duplicate dataset : {}/{}".format(len(src_dedup_df), len(df[~df.isin(src_dedup_df)].dropna())))

#########################################################################
#Substring freq dictionary
#########################################################################

#create substring:freq dictionary
substring_frequencies = {}
for substring_list in src_dedup_df["src_substring_ngram_segments"].tolist():
    for substring in substring_list:
        if substring not in substring_frequencies.keys():
            substring_frequencies[substring]=1
        else:
            substring_frequencies[substring]=substring_frequencies[substring]+1
print("Total number of substring_frequencies.keys() : {}".format(len(substring_frequencies)))

#Unique substrings
print("Total number of Unique substrings : {}".format(len(substring_frequencies.keys())))

#create indexes for substring
indexed_substrings = {}
for index, substring in enumerate(substring_frequencies.keys()):
            indexed_substrings[substring]= index
assert  len(substring_frequencies.keys()) == len(indexed_substrings)

#convert substrings to indexes
def substring2index(substrings):
    return set([int(indexed_substrings[str(substring)]) for substring in substrings])

#sentences - Overlapping/Non-Overlapping/Short-Sents/Long-Sents
def get_non_ovelaping_sentences(row):
    if len(row["src_sent_words"]) > 40:
        return "Long-Sents"
    elif len(row["src_substring_ngram_segments"]) > 0:
        overlaps = [True for substring in row["src_substring_ngram_segments"] if substring_frequencies[substring] > 1]
        if any(overlaps):
            return "Overlapping"
        else:
            return "Non-overlapping"
    return "Short-Sents"

#overlapping/Non-overlapping/Short-Sents/Long-Sents
src_dedup_df["non_overlapping"]= src_dedup_df.apply(get_non_ovelaping_sentences, axis=1)

###################################################################
#pre-processing
##################################################################
#drop short sentences than tgr_ngram_segments
print("\n\nFiltering unique sentences:")
print("-------------------------------")
preprocessed_df = src_dedup_df[src_dedup_df["non_overlapping"] == "Overlapping"]
print("Dataframe size after valid/Short-Sents/Long-Sents/Unique_sentences : {}/{}/{}/{}".format(len(preprocessed_df), len(src_dedup_df[src_dedup_df["non_overlapping"] == "Short-Sents"]), len(src_dedup_df[src_dedup_df["non_overlapping"] == "Long-Sents"]), len(src_dedup_df[src_dedup_df["non_overlapping"] == "Non-overlapping"])))
print("Preprocessed columns : {}".format(preprocessed_df.columns.tolist()))
print(preprocessed_df[:5])

endTime = time.time()
print('Time taken to complete src ngram_segments: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))

#reindexing
# reindexed_src_initial_df.reset_index(drop=True, inplace=True)
# current_src_df = reindexed_src_initial_df.iloc[:,:]
# using dask https://stackoverflow.com/questions/45545110/make-pandas-dataframe-apply-use-all-cores
# current_src_df = dd.from_pandas(reindexed_src_initial_df.iloc[:,:], npartitions=30)

###################################################################
#Main Logic
###################################################################

mapply.init(
    n_workers=1,
    chunk_size=1,
    max_chunks_per_worker=0,
    progressbar=False,
)

def return_indexes(row, current_preprocessed_df):
    #current_preprocessed_df["src_char_difference"] = current_preprocessed_df.apply(lambda current_row_selected : len(set(row["src_sent_chars"]) - set(current_row_selected["src_sent_chars"])), axis=1)
    #subset_current_src_df = current_preprocessed_df[current_preprocessed_df["src_sent_chars_length"] < row["src_sent_chars_length"]+sliding_window_size].iloc[:,:]
    current_preprocessed_df["Levenshtien_distance"] = current_preprocessed_df["src_mod_sents"].apply(lambda current_row_selected: distance(str(row["src_mod_sents"]), str(current_row_selected)))
    filtered_current_df = current_preprocessed_df[current_preprocessed_df["Levenshtien_distance"] <= levenshtein_distance_threshold]
    print("Index [{}] of rows in the current subset_current_src_df : {}".format(row.name, len(current_preprocessed_df)))
    return list(set(filtered_current_df.index.tolist())-{row.name})

###########################################################################
###main filtration
###########################################################################

#create copy to process
preprocessed_copy_df = preprocessed_df.iloc[:,:]
print("No of rows in the preprocessed_copy_df : {}".format(len(preprocessed_copy_df)))

#calculate values for reindexed_filtered_simSents
preprocessed_df["src_filtered_indexes"] = preprocessed_df.mapply(lambda row: return_indexes(row, preprocessed_copy_df.copy()), axis=1)

#write-into files
df.drop(columns=["src_mod_sents", "src_sent_words", "src_sent_chars_length", "src_substring_ngram_segments"], axis=1, inplace=True)
#simSents to filter
filtered_src_sents_indexes = list(set([i for index_list in preprocessed_df["src_filtered_indexes"].tolist() for i in index_list]))
print("No of filtered_src_sents_indexes : {}".format(len(filtered_src_sents_indexes)))
filtered_src_sents_df = df.loc[filtered_src_sents_indexes]
filtered_src_sents_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, src_similar_sents_csv_file), index=False)

src_ref_sentences_indexes = list(set(preprocessed_df[preprocessed_df['src_filtered_indexes'].apply(lambda x: len(x) > 0)].index.tolist()))
src_ref_sents_df = df.loc[src_ref_sentences_indexes]
print("No of ref_src_sents_df : {}".format(len(src_ref_sents_df)))
src_ref_sents_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, ref_src_sents_csv_file), index=False)


#After filtration - remove duplicate_indexes from org
print("\nPost Processing:")
print("--------")
print("Original Dataframe size : {}".format(len(df)))
df.drop(index=df[~df.isin(src_dedup_df)].dropna().index.tolist(), axis=1, inplace=True)
print("After dropping duplicates from org dataset : {}".format(len(df)))

#remove filtered indexes
print("Total filtered_src_sents_indexes for filtering : {}".format(len(filtered_src_sents_indexes)))
df.drop(index=filtered_src_sents_indexes, axis=1, inplace=True)
print("Final df size after removing filtered indexes : {}".format(len(df)))
endTime = time.time()
print('Time taken to complete Script: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))

#laser3_scores sorted
laser3_src_dedup_sorted_df= df.sort_values("laser3_scores", ascending=False)
laser3_src_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_src_sorted_csv_file), index=False)

#xlmr_scores sorted
xlmr_src_dedup_sorted_df= df.sort_values("xlmr_scores", ascending=False)
xlmr_src_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_src_sorted_csv_file), index=False)

#labse_scores sorted
labse_src_dedup_sorted_df= df.sort_values("labse_scores", ascending=False)
labse_src_dedup_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_src_sorted_csv_file), index=False)

endTime = time.time()
print('Time taken to complete Script: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print("Script Completed.....")
