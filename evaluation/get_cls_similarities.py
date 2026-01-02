import os
import argparse
from msclap import CLAP
import pickle
import torch


import os
import argparse
from config import conf, common_parameters
from pprint import pprint
from utilities import merge_dicts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Training script for learning promps for audio-text models"
    parser.add_argument("--conf_id", required=True,
                        help="Configuration tag located\
                            in config.py for the experiment")
    parser.add_argument("--job_id", default=None,
                        help="ID of the job to run. If not provided,\
                              a new ID job will be created")
    
    args = parser.parse_args()
    return args


def main(conf):

    model_name = conf['model_name']
    test_dataset = conf['test_dataset']
    definition_type = conf['definition_type']
    model_name = conf['model_name']
    evaluation_mode = conf['evaluation_mode']

    # Load dataset
    audio_embeddings_path = os.path.join(conf['audio_embeddings_folder'],
                                            model_name,
                                            test_dataset + '.pt')
    text_embeddings_path = os.path.join(conf['text_embeddings_folder'],
                                        model_name,
                                        test_dataset + '_' + definition_type + '.pkl')
    

    
    
    # Load CLAP model
    if model_name == 'CLAP-MS-23':
        clap_model = CLAP(version = '2023', use_cuda=True)
    elif model_name == 'CLAP-MS-22':
        clap_model = CLAP(version = '2022', use_cuda=True)
    elif model_name == 'CLAP-LA':
        pass
    else:
        raise ValueError('Please specify a valid model')


    
    # Read embeddings
    audio_embeddings = torch.load(audio_embeddings_path)
    print("Audio embeddings shape: ", audio_embeddings.shape)

    # Read ground-truth labels
    labels = torch.load(audio_embeddings_path.replace('.pt', '_labels.pt'))
    print("Labels shape: ", labels.shape)

    # Read text embeddings
    with open(text_embeddings_path, 'rb') as f:
        prompts_dictionary = pickle.load(f)

    
    # Compose prompt template for evaluation
    if evaluation_mode == 'CLS':
        prompt_template = ''
    elif evaluation_mode == 'baseline':
        prompt_template = 'This is a sound of '

    if evaluation_mode == 'CLS' or evaluation_mode == 'baseline':
        
        text_embeddings = prompts_dictionary[prompt_template]['embeddings']
        print("Text embeddings shape: ", text_embeddings.shape)

        # Compute similarity
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        print("Similarity shape: ", similarity.shape)

        # Save similarity tensors to disk as pt files
        output_folder = os.path.join(conf['similarities_folder'], model_name)
        os.makedirs(output_folder, exist_ok=True)
        similarity_path = os.path.join(output_folder, test_dataset + '_' + definition_type + '.pt')
        torch.save(similarity, similarity_path)

    
    else:
        raise ValueError('Please specify a valid evaluation mode')

    


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    args = vars(args)
    print('Input arguments: ', args)

    conf = merge_dicts(common_parameters, conf[args["conf_id"]])
    conf = {**conf, **args}
    
    pprint(conf)


    main(conf)
