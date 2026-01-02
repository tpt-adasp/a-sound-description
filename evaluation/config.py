'''
Configuration file to evaluate CLAP on Zero-shot classification tasks
using different class definitions in a cross-validation setup
'''


import os
import random
import string

TASK = "dcase"
STUDY = "016_CLAP_prompting_with_descriptors"
EXP = "003_evaluate_prompts"

slurm = "SLURM_JOB_ID" in os.environ # check if running on slurm
if slurm:
    JOB_ID = os.environ["SLURM_JOB_ID"] # get job idcd
    print("Running on slurm with job id:", JOB_ID)
else:
    JOB_ID = "".join(random.choices(
        string.ascii_letters + string.digits, k=8)) # generate random job id


studies_folder = os.environ["STUDIES"]
studies_data_folder = os.environ["STUDIES_DATA"]

output_folder = os.path.join(studies_data_folder,
                             STUDY,
                             EXP,
                             "results")

similarities_folder = os.path.join(studies_data_folder,
                             STUDY,
                             EXP,
                             "similarities")

definitions_folder = os.path.join(studies_data_folder,
                                  '013_caption_generation',
                                  '001_definitions')

base_definitions_path = os.path.join(definitions_folder,
                                     "baseline_definitions-Mistral-auto-3-300-dataset_label.tsv")

context_definitions_path = os.path.join(definitions_folder,
                                        "context_aware_definitions-Mistral-auto-3-300-dataset_label.tsv")

ontology_definitions_path = os.path.join(definitions_folder,
                                         "ontology_aware_definitions-Mistral-auto-3-300-dataset_label.tsv")
                                         


audio_embeddings_folder = os.path.join(studies_data_folder,
                                             STUDY,
                                             '001_extract_audio_embeddings',
                                             'embeddings')

text_embeddings_folder = os.path.join(studies_data_folder,
                                                STUDY,
                                                '002_extract_text_embeddings',
                                                'embeddings')
                                        
                                  


common_parameters = {
    "job_id": JOB_ID,
    'output_folder': output_folder,
    'similarities_folder': similarities_folder,
    'audio_embeddings_folder': audio_embeddings_folder,
    'text_embeddings_folder': text_embeddings_folder



}

conf = {

     "001": {
        'model_name': "CLAP-MS-23",
        'definition_type': "CLS",
        'test_dataset': "ESC50", # ESC50, US8K, TUT2017, AudioSet, DCASE17, FSD50K
        'evaluation_mode': 'CLS' # CLS, baseline, ensemble, 
     },

     "002": {
        'model_name': "CLAP-MS-23",
        'definition_type': "CLS",
        'test_dataset': "FSD50K", # ESC50, US8K, TUT2017, AudioSet, DCASE17, FSD50K
        'evaluation_mode': 'CLS' # CLS, baseline, ensemble, 
     },

     
}
