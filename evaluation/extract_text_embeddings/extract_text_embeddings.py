import os
import json
import argparse
import pandas as pd
from msclap import CLAP
from tqdm import tqdm
import pickle


import os
import argparse
import json
from config import conf, common_parameters
from pprint import pprint
from utilities import merge_dicts

from ESC50_dataset import ESC50
from FSD50K_dataset import FSD50K


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Script to extract text embeddings"
    parser.add_argument("--conf_id", required=True,
                        help="Configuration tag located\
                            in config.py for the experiment")
    parser.add_argument("--job_id", default=None,
                        help="ID of the job to run. If not provided,\
                              a new ID job will be created")
    
    args = parser.parse_args()
    return args


def main(conf):


    dataset_name = conf['dataset']
    model_name = conf['model_name']
    definition_type = conf['definition_type']

    # Load dataset
    if dataset_name =='ESC50':
        root_path = os.path.join(os.environ["DCASE_DATA"], "ESC-50v2")
        dataset = ESC50(root=root_path, download=False)

    elif dataset_name == 'FSD50K':
        root_path = os.path.os.environ["DCASE_DATA"]
        dataset = FSD50K(root=root_path, download=False)
    else:
        raise ValueError('Please specify a valid dataset')


    # Load CLAP model
    if model_name == 'CLAP-MS-23':
        clap_model = CLAP(version = '2023', use_cuda=True)
    elif model_name == 'CLAP-MS-22':
        clap_model = CLAP(version = '2022', use_cuda=True)
    elif model_name == 'CLAP-LA':
        pass
    else:
        raise ValueError('Please specify a valid model')


    # Load prompts
    
    # Read json file with prompt templates
    with open('prompt_templates.json', 'r') as f:
        prompts = json.load(f)

    prompt_templates = prompts['prompt_templates']
    print("Prompt templates: ", prompt_templates)

    # Load class descriptions

    if definition_type == 'CLS':
        # No class descriptions needed
        pass
    elif definition_type == 'dictionary':
        # Load class descriptions from Cambridge dictionary
       definitions_filepath = conf['dictionary_definitions_path']

    elif definition_type == 'base':
        # Load base audio-centric class descriptions from Mistral
        if dataset_name == 'TUT2017':
            definitions_filepath = conf['asc_base_definitions_path']
        else:
            definitions_filepath = conf['base_definitions_path']

    elif definition_type == 'ontology':
        # Load ontology-aware class descriptions from Mistral
        if dataset_name == 'TUT2017':
            definitions_filepath = conf['asc_ontology_definitions_path']
        else:
            definitions_filepath = conf['ontology_definitions_path']

    elif definition_type == 'context':
        # Load context-aware class descriptions from Mistral
        definitions_filepath = conf['context_definitions_path']
    else:
        raise ValueError('Please specify a valid definition type')
    

    if definition_type != 'CLS':
        # CLass descriptions datraframe
        class_descriptions = pd.read_csv(os.path.join(conf['definitions_folder'],
                                                    definitions_filepath), sep='\t')
        # Make definition column start with a lower case letter
        class_descriptions['definition'] = class_descriptions['definition'].apply(lambda x: str(x)[0].upper() + str(x)[1:])

        print(class_descriptions.head())


    # Compose a dictionary with the prompt templates for each class
    # Key corresponds to the prompt template and Value is a dictionary
    # with keys 'textual_description' and 'embedding'
    prompt_dict = {}

    # First, check that dataset classes are in the class descriptions
    # print list of classes that are not in the class descriptions
    print("Checking availability of class descriptions for dataset classes")
    if definition_type != 'CLS':
        counter = 0
        for x in dataset.classes:
            if dataset_name == 'FSD50K':
                if x.replace(' ', '_') not in class_descriptions['label'].values:
                    print(x)
                counter += 1
            elif dataset_name == 'AudioSet' or dataset_name == 'DCASE2017':
                if x not in class_descriptions['label'].values:
                    print(x)
                    counter += 1
        print("Number of classes without class descriptions: ", counter)
    else:
        pass
    

    # Dataset classes start with a lower case letter
    dataset.classes = [str(x)[0].lower() + str(x)[1:] for x in dataset.classes]
    print("Dataset classes: ", dataset.classes)


    # Get text embeddings
    for template in tqdm(prompt_templates):
        # print("Prompt: ", template)
        # Dataset classes start with a lower case letter
        if dataset_name == 'FSD50K' or dataset_name =='AudioSet' or dataset_name == 'DCASE2017':
            dataset.classes = [str(x)[0].lower() + str(x)[1:] for x in dataset.classes]
        y = [template + x for x in dataset.classes]
        # Make sure that the first letter of the sentence is capitalized
        y = [x[0].upper() + x[1:] for x in y]

        # Revert to original class names
        if dataset_name == 'FSD50K' or dataset_name == 'AudioSet' or dataset_name == 'DCASE2017':
            dataset.classes = [str(x)[0].upper() + str(x)[1:] for x in dataset.classes]
        if definition_type != 'CLS':
            # Add class descriptions to the textual descriptions
            if dataset_name == 'FSD50K' or dataset_name == 'ESC50' or dataset_name == 'US8K' or dataset_name == 'TUT2017':
                # Sanity check. Print class descriptions for all classes
                # for x in dataset.classes:
                #     print(x, class_descriptions[class_descriptions['label'] == x.replace(' ', '_')]['definition'].values[0])
                y = [y + '. ' + class_descriptions[class_descriptions['label'] == x.replace(' ', '_')]['definition'].values[0] for x, y in zip(dataset.classes, y)]
            elif dataset_name == 'AudioSet' or dataset_name == 'DCASE2017':
                y = [y + '. ' + class_descriptions[class_descriptions['label'] == x]['definition'].values[0] for x, y in zip(dataset.classes, y)]
        else:
            y = y
            # Make sure that the first letter of the sentence is capitalized
            y = [x[0].upper() + x[1:] for x in y]
        # Add a dot at the end of the sentence only if it is not already there
        y = [x + '.' if x[-1] != '.' else x for x in y]
        print("Textual descriptions: ", y)
        text_embeddings = clap_model.get_text_embeddings(y)
        prompt_dict[template] = {'textual_descriptions': y,
                               'embeddings': text_embeddings}
        
    
    for i, template in enumerate(y):
        print(i, template)


    

    # Save dictionary to a pickle file
    output_folder = os.path.join(conf['output_folder'], model_name)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, dataset_name + '_' + definition_type +  '.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(prompt_dict, f)

   



if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    args = vars(args)
    print('Input arguments: ', args)

    conf = merge_dicts(common_parameters, conf[args["conf_id"]])
    conf = {**conf, **args}
    
    pprint(conf)


    main(conf)
