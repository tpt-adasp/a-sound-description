'''
Configuration file to extract CLAP audio embeddings
'''


import os
import random
import string

TASK = "dcase"
STUDY = "016_CLAP_prompting_with_descriptors"
EXP = "002_extract_text_embeddings"

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
                             "embeddings")

definitions_folder = os.path.join(studies_data_folder,
                                  '013_caption_generation',
                                  '001_definitions_extended_for_DCASE')

base_definitions_path = os.path.join(definitions_folder,
                                     "baseline_definitions-Mistral-auto-3-300-dataset_label.tsv")

context_definitions_path = os.path.join(definitions_folder,
                                        "context_aware_definitions-Mistral-auto-3-300-dataset_label.tsv")

ontology_definitions_path = os.path.join(definitions_folder,
                                         "ontology_aware_definitions-Mistral-auto-3-300-dataset_label.tsv")
dictionary_definitions_path = os.path.join(definitions_folder,
                                           "dictionary_dataset_labels_definitions.tsv")

asc_base_definitions_path = os.path.join(definitions_folder,
                                           "asc-baseline_definitions-Mistral-auto-3-300-dataset_label.tsv")
                                         
asc_ontology_definitions_path = os.path.join(definitions_folder,
                                             "asc-ontology_aware_definitions-Mistral-auto-3-300-dataset_label.tsv")
                                                                                     
                                  
common_parameters = {
    "job_id": JOB_ID,
    'output_folder': output_folder, 
    'definitions_folder': definitions_folder,
    'base_definitions_path': base_definitions_path,
    'context_definitions_path': context_definitions_path,
    'ontology_definitions_path': ontology_definitions_path,
    'dictionary_definitions_path': dictionary_definitions_path,
    'asc_base_definitions_path': asc_base_definitions_path,
    'asc_ontology_definitions_path': asc_ontology_definitions_path,


}

conf = {
    # ESC50 dataset
     "001": {
        "model_name": "CLAP-MS-23",
        "dataset": "ESC50",
        "definition_type": "context", # CLS, dictionary, base, context, ontology
     },
     "002": {
        "model_name": "CLAP-MS-23",
        "dataset": "ESC50",
        "definition_type": "ontology",
     },
     "003": {
        "model_name": "CLAP-MS-23",
        "dataset": "ESC50",
        "definition_type": "base", 
     },
     "004": {
        "model_name": "CLAP-MS-23",
        "dataset": "ESC50",
        "definition_type": "CLS", 
     },
     "005": {
        "model_name": "CLAP-MS-23",
        "dataset": "ESC50",
        "definition_type": "dictionary", 
     },
     
     # FSD50K dataset
     "021": {
        "model_name": "CLAP-MS-23",
        "dataset": "FSD50K",
        "definition_type": "context", 
     },
     "022": {
        "model_name": "CLAP-MS-23",
        "dataset": "FSD50K",
        "definition_type": "ontology", 
     },
     "023": {
        "model_name": "CLAP-MS-23",
        "dataset": "FSD50K",
        "definition_type": "base", 
     },
     "024": {
        "model_name": "CLAP-MS-23",
        "dataset": "FSD50K",
        "definition_type": "CLS", 
     },
     "025": {
        "model_name": "CLAP-MS-23",
        "dataset": "FSD50K",
        "definition_type": "dictionary",
     },
     
}

