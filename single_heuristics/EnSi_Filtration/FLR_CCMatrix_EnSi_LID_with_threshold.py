# Created by  aloka on 23/03/2024

#imports
import os
import time
import pandas as pd
import fasttext
from datasets import load_dataset
#from laser_encoders import LaserEncoderPipeline
#from sentence_transformers import SentenceTransformer, util


startTime=time.time()
#inputs
data_dir="/userdirs/aloka/datasets/opus/CCMatrix/en-si.txt"
input_file = "CCMatrix-scores-sample-1M-en-si.csv"
#input_file= open("{}/{}".format(data_dir, input_file), "r", encoding="utf8")

#LID model
model = fasttext.load_model("/userdirs/aloka/uom_google_project/nllblid218e")

#outputs
filtration_exp = "LID_threshold"
LID_prob_threshold = 0.7
src_lang="en"
tgt_lang="si"

#output files
sub_dir = "csv_files"
output_dir="/userdirs/aloka/p4_parallel_data_curation/data/ccmatrix_{}_exp_sample1M".format(filtration_exp)
if not os.path.isdir("{}/{}".format(output_dir, sub_dir)):
    os.makedirs("{}/{}".format(output_dir, sub_dir))
else:
    print("Directory Exists...........")


#output files
sub_dir = "csv_files"
laser3_sorted_src_slength_csv_file="{}-{}-laser3-sorted-{}-{}.train.csv".format(src_lang, tgt_lang, src_lang, filtration_exp)
xlmr_sorted_src_slength_csv_file="{}-{}-xlmr-sorted-{}-{}.train.csv".format(src_lang, tgt_lang, src_lang, filtration_exp)
labse_sorted_src_slength_csv_file="{}-{}-labse-sorted-{}-{}.train.csv".format(src_lang, tgt_lang, src_lang, filtration_exp)

laser3_sorted_tgt_slength_csv_file="{}-{}-laser3-sorted-{}-{}.train.csv".format(src_lang, tgt_lang, tgt_lang, filtration_exp)
xlmr_sorted_tgt_slength_csv_file="{}-{}-xlmr-sorted-{}-{}.train.csv".format(src_lang, tgt_lang, tgt_lang, filtration_exp)
labse_sorted_tgt_slength_csv_file="{}-{}-labse-sorted-{}-{}.train.csv".format(src_lang, tgt_lang, tgt_lang, filtration_exp)

laser3_sorted_src_tgt_slength_csv_file="{}-{}-laser3-sorted-{}-{}-{}.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang, filtration_exp)
xlmr_sorted_src_tgt_slength_csv_file="{}-{}-xlmr-sorted-{}-{}-{}.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang, filtration_exp)
labse_sorted_src_tgt_slength_csv_file="{}-{}-labse-sorted-{}-{}-{}.train.csv".format(src_lang, tgt_lang, src_lang, tgt_lang, filtration_exp)


#predict LID
def get_lid(text):
    try:
        predictions = model.predict(str(text), k=1)
        lang_code = predictions[0][0].strip().split('__')[-1]
        prob = predictions[1][0]

        #return lang_code, prob
        return  pd.Series([lang_code, prob])
    except:
        return pd.Series(["UNK", 0.0])


###############################################################
#main logic
###############################################################

#read data
org_dataset_df=pd.read_csv("{}/{}".format(data_dir, input_file), delimiter=",", encoding="utf8")
print('Dataset size original : {}'.format(len(org_dataset_df)))

#with LID columns
org_dataset_df[["src_LID", "src_LID_prob"]] = org_dataset_df["src_sents"].apply(get_lid)
org_dataset_df[["tgt_LID", "tgt_LID_prob"]] = org_dataset_df["tgt_sents"].apply(get_lid)
print(org_dataset_df.head(5))
print('Entries with Src UNK : \n'.format(org_dataset_df.loc[org_dataset_df['src_LID'] =="UNK"]))
print('Entries with Tgt UNK : \n'.format(org_dataset_df.loc[org_dataset_df['tgt_LID'] =="UNK"]))


#LID filtration
src_LID_df = org_dataset_df.loc[(org_dataset_df['src_LID'] =="eng_Latn") & (org_dataset_df['src_LID_prob'] > LID_prob_threshold)]
print('Dataset size after src LID filtering : {}'.format(len(src_LID_df)))
tgt_LID_df = org_dataset_df.loc[(org_dataset_df['tgt_LID'] =="sin_Sinh") & (org_dataset_df['tgt_LID_prob'] > LID_prob_threshold)]
print('Dataset size after tgt LID filtering : {}'.format(len(tgt_LID_df)))
src_tgt_LID_df = org_dataset_df.loc[(org_dataset_df['src_LID'] =="eng_Latn") & (org_dataset_df['src_LID_prob'] > LID_prob_threshold) & (org_dataset_df['tgt_LID'] =="sin_Sinh") & (org_dataset_df['tgt_LID_prob'] > LID_prob_threshold)]
print('Dataset size after src & tgt LID  filtering : {}'.format(len(src_tgt_LID_df)))

#laser3_scores sorted
laser3_src_slength_sorted_df= src_LID_df.sort_values("laser3_scores", ascending=False)
laser3_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_slength_csv_file), index=False)
laser3_tgt_slength_sorted_df= tgt_LID_df.sort_values("laser3_scores", ascending=False)
laser3_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_tgt_slength_csv_file), index=False)
laser3_src_tgt_slength_sorted_df= src_tgt_LID_df.sort_values("laser3_scores", ascending=False)
laser3_src_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, laser3_sorted_src_tgt_slength_csv_file), index=False)

#xlmr_scores sorted
xlmr_src_slength_sorted_df= src_LID_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_slength_csv_file), index=False)
xlmr_tgt_slength_sorted_df= tgt_LID_df.sort_values("xlmr_scores", ascending=False)
xlmr_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_tgt_slength_csv_file), index=False)
xlmr_src_tgt_slength_sorted_df= src_tgt_LID_df.sort_values("xlmr_scores", ascending=False)
xlmr_src_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, xlmr_sorted_src_tgt_slength_csv_file), index=False)

#labse_scores sorted
labse_src_slength_sorted_df= src_LID_df.sort_values("labse_scores", ascending=False)
labse_src_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_slength_csv_file), index=False)
labse_tgt_slength_sorted_df= tgt_LID_df.sort_values("labse_scores", ascending=False)
labse_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_tgt_slength_csv_file), index=False)
labse_src_tgt_slength_sorted_df= src_tgt_LID_df.sort_values("labse_scores", ascending=False)
labse_src_tgt_slength_sorted_df.to_csv("{}/{}/{}".format(output_dir, sub_dir, labse_sorted_src_tgt_slength_csv_file), index=False)

endTime = time.time()
print('Time taken to complete LID filtration: {}min {}sec'.format(int((endTime - startTime) // 60), int((endTime - startTime) % 60)))
print('----------------------------------------------------------------------------------------------------------------\n\n')
print("Script Completed.....")