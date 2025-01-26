import owlready2
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline_from_config
from pykeen.hpo import hpo_pipeline
import pandas as pd
import torch
import os
import json
import pickle
import wandb


def train_embeddings(pykeen_data, model_name='RotatE', dev_mode=False, dev_samples=1000):
    """
    Train knowledge graph embeddings using PyKEEN pipeline.
    """
    # Initialize wandb run
    # wandb.init(
    #     project="knowledge-graph-embeddings",
    #     config={
    #         "model": model_name,
    #         "dev_mode": dev_mode,
    #         "dev_samples": dev_samples if dev_mode else "full",
    #         "batch_size": 128,
    #         "epochs": 1 if dev_mode else 100,
    #     }
    # )

    if dev_mode:
        # Instead of using from_labeled_triples, just slice the existing TriplesFactory
        training = TriplesFactory(
            mapped_triples=pykeen_data['train'].mapped_triples[:dev_samples],
            entity_to_id=pykeen_data['train'].entity_to_id,
            relation_to_id=pykeen_data['train'].relation_to_id
        )
        validation = TriplesFactory(
            mapped_triples=pykeen_data['validation'].mapped_triples[:dev_samples//10],
            entity_to_id=pykeen_data['validation'].entity_to_id,
            relation_to_id=pykeen_data['validation'].relation_to_id
        )
        testing = TriplesFactory(
            mapped_triples=pykeen_data['test'].mapped_triples[:dev_samples//10],
            entity_to_id=pykeen_data['test'].entity_to_id,
            relation_to_id=pykeen_data['test'].relation_to_id
        )
    else:
        training = pykeen_data['train']
        validation = pykeen_data['validation']
        testing = pykeen_data['test']

    training_kwargs = {
        'batch_size': 256,
        'num_epochs': 1 if dev_mode else 100,
    }

    # Get the best trial's model from the HPO pipeline
    result = hpo_pipeline(
        model=model_name,
        training=training,
        testing=testing,
        validation=validation,
        training_kwargs=training_kwargs,
        result_tracker='wandb',
        result_tracker_kwargs={'project': 'knowledge-graph-embeddings'},
        evaluation_kwargs=dict(batch_size=256),
        epochs=1 if dev_mode else 10,
        n_trials=1 if dev_mode else 30,
    )

    # Log best trial metrics
    # wandb.log({
    #     "best_trial_score": result.study.best_trial.value,
    #     "best_trial_params": result.study.best_trial.params
    # })

    print(result.study)
    result.save_to_directory('ncit/ncit_embeddings2')
    
    # load the best pipeline
    with open('ncit/ncit_embeddings2/best_pipeline/pipeline_config.json', 'r') as f:
        pipeline_config = json.load(f)

    pipeline_config['pipeline']['training'] = training
    pipeline_config['pipeline']['testing'] = testing
    pipeline_config['pipeline']['validation'] = validation
    pipeline_config['pipeline']['training_kwargs']['num_epochs'] = 1 if dev_mode else 100
    pipeline_config['pipeline']['stopper'] = 'early'
    pipeline_config['pipeline']['stopper_kwargs'] = {'patience': 10, 'relative_delta': 0.01}

    best_pipeline_result = pipeline_from_config(pipeline_config)
    
    # Log final model metrics
    # metrics = best_pipeline_result.metric_results.to_dict()
    # wandb.log({"final_metrics": metrics})
    
    # wandb.finish()
    return best_pipeline_result

def main(dataset_path, model_name='RotatE', dev_mode=False):
    """
    Complete pipeline from OWL file to trained embeddings.
    """

    with open(dataset_path, 'rb') as f:
        pykeen_data = pickle.load(f)
                            
    # Train embeddings
    print(f"Training {model_name} embeddings...")
    result = train_embeddings(pykeen_data, model_name, dev_mode=dev_mode, dev_samples=1000)

    print(result)
    
    # Get embeddings
    entity_embeddings = result.model.entity_representations[0]()
    relation_embeddings = result.model.relation_representations[0]()

    print(entity_embeddings)
    
    return {
        'result': result,
        'entity_embeddings': entity_embeddings,
        'relation_embeddings': relation_embeddings,
        'entity_to_id': result.training.entity_to_id,
        'relation_to_id': result.training.relation_to_id
    }


if __name__ == "__main__":
    # Set your wandb API key (alternatively use environment variable WANDB_API_KEY)
    # wandb.login(key="your-api-key")
    
    result = main(dataset_path='ncit/ncit_embeddings/pykeen_data.pkl',
         model_name='PairRE',
         dev_mode=False)
    
    # Save the embeddings and mappings
    print("Saving embeddings and mappings...")
    save_dir = 'ncit/ncit_embeddings2'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save entity embeddings tensor
    torch.save(result['entity_embeddings'], os.path.join(save_dir, 'entity_embeddings.pt'))
    
    # Save relation embeddings tensor  
    torch.save(result['relation_embeddings'], os.path.join(save_dir, 'relation_embeddings.pt'))
    
    # Save entity and relation mappings
    pd.DataFrame.from_dict(result['entity_to_id'], orient='index', columns=['id']).to_csv(
        os.path.join(save_dir, 'entity_to_id.csv'))
        
    pd.DataFrame.from_dict(result['relation_to_id'], orient='index', columns=['id']).to_csv(
        os.path.join(save_dir, 'relation_to_id.csv'))
    
    # Save full training result object
    # torch.save(result['result'], os.path.join(save_dir, 'training_result.pt'))
    
    print(f"Saved embeddings and mappings to {save_dir}/")
         