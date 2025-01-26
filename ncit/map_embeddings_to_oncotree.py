import pandas as pd
import os
import json
import hydra
from omegaconf import DictConfig
from owlready2 import get_ontology
import torch
manual_entity_mappings = {"tp53_pathway": "C91572",
"rtk_ras_pathway": "C39217",
"wnt_pathway": "C91531",
"myc_pathway": "C38346",
"pi3k_pathway": "C39204",
"notch_pathway": "C91520",
"tgf_beta_pathway": "C39251",
"msi_type": "C36318",
"impact_tmb_score": "C150128",
"COAD-msi_type.0": "C173324",
"COAD-msi_type.1": "C162256", 
"UEC.msi_type.0": "C180335",
"UEC.msi_type.1": "C180514",
"IDC-HR.1-HER2.1": "C53555",
"IDC-HR.1-HER2.0": "C53554",
"IDC-HR.0-HER2.1": "C53556",
"IDC-HR.0-HER2.0": "C53558",
"HGSOC": "C105555",
"BRCANOS": "C4872"
        }

def load_entity_names():
    cache_path = os.path.join(hydra.utils.get_original_cwd(), 'ncit/entity_names.json')
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    else:
        onco_path = os.path.join(hydra.utils.get_original_cwd(), 'ncit/ThesaurusInferred.owl')
        onto = get_ontology(onco_path).load()
        entity_names = {}
        for c in onto.classes():
            code = c.iri.split('#')[-1]
            if hasattr(c, 'label') and len(c.label) > 0:
                entity_names[code] = c.label[0]
        with open(cache_path, 'w') as f:
            json.dump(entity_names, f)
        return entity_names

def map_oncotree_to_ncit(entity_names, cfg):
    # Load entity to index mapping
    entity_to_index_df = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'ncit/ncit_embeddings2/entity_to_id.csv'))
    entity_to_index_df.columns = ['entity', 'index']
    
    # Add names to the dataframe
    entity_to_index_df['name'] = entity_to_index_df['entity'].map(entity_names)
    
    print(entity_to_index_df.head())

    # load oncotree mapping
    oncotree_mapping = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'ncit/ncit_embeddings/ontology_mappings.txt'), sep='\t')[['NCIT_CODE', 'ONCOTREE_CODE']]
    print(oncotree_mapping.head())

    entity_to_oncotree_df = entity_to_index_df.merge(oncotree_mapping, left_on='entity', right_on='NCIT_CODE', how='inner')

    print(entity_to_oncotree_df)

    # save entity to oncotree mapping
    entity_to_oncotree_df.to_csv(os.path.join(hydra.utils.get_original_cwd(), 'ncit/ncit_embeddings2/entity_to_id_with_oncotree.csv'), index=False)


def map_targets_to_ncit(entity_names, cfg):
    inverse_entity_names = {v: k for k, v in entity_names.items()}

    # load targets
    with open(os.path.join(hydra.utils.get_original_cwd(), cfg.gam.target_list_path), 'r') as f:
        target_dicts = json.load(f)

    target_mapping = {"gene_symbol": [], "entity_name": []}
    target_names = [x['target'] for x in target_dicts]
    unmatched_targets = []
    for target_name in target_names:
        if f"{target_name} Gene" in inverse_entity_names.keys():
            target_mapping["gene_symbol"].append(target_name)
            target_mapping["entity_name"].append(inverse_entity_names[f"{target_name} Gene"])
        elif target_name in manual_entity_mappings.keys():
            target_mapping["gene_symbol"].append(target_name)
            target_mapping["entity_name"].append(manual_entity_mappings[target_name])
        else:
            print(f"Target {target_name} not found in entity names")
            unmatched_targets.append(target_name)
            continue

    print("Unmatched targets:")
    for target in unmatched_targets:
        print(f'"{target}": "",')
    target_mapping = pd.DataFrame(target_mapping)

    entity_to_index_df = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'ncit/ncit_embeddings2/entity_to_id.csv'))
    entity_to_index_df.columns = ['entity', 'index']
    target_mapping = target_mapping.merge(entity_to_index_df, left_on='entity_name', right_on='entity', how='inner')
    target_mapping = target_mapping[['gene_symbol', 'entity', 'index']]
    target_mapping.to_csv(os.path.join(hydra.utils.get_original_cwd(), 'ncit/ncit_embeddings2/entity_to_id_with_target.csv'), index=False)


def save_oncotree_embeddings(path):
    embeddings = torch.load(path)
    mapping = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'ncit/ncit_embeddings2/entity_to_id_with_oncotree.csv'))
    # make dictionary and save to json
    mapping_dict = {row['ONCOTREE_CODE']: embeddings[row['index'], :].tolist() for _, row in mapping.iterrows()}
    with open(os.path.join(hydra.utils.get_original_cwd(), '../data-commons/oncotree_embeddings.json'), 'w') as f:
        json.dump(mapping_dict, f)

def save_target_embeddings(path):
    embeddings = torch.load(path)
    mapping = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'ncit/ncit_embeddings2/entity_to_id_with_target.csv'))
    # make dictionary and save to json
    mapping_dict = {row['gene_symbol']: embeddings[row['index'], :].tolist() for _, row in mapping.iterrows()}
    with open(os.path.join(hydra.utils.get_original_cwd(), '../data-commons/target_embeddings.json'), 'w') as f:
        json.dump(mapping_dict, f)

@hydra.main(config_path='../config', config_name='defaults')
def main(cfg: DictConfig):    
    # Create dictionary of entity codes to names
    entity_names = load_entity_names()

    map_oncotree_to_ncit(entity_names, cfg)
    map_targets_to_ncit(entity_names, cfg)

    save_oncotree_embeddings(os.path.join(hydra.utils.get_original_cwd(), "ncit/ncit_embeddings2/entity_embeddings.pt"))
    save_target_embeddings(os.path.join(hydra.utils.get_original_cwd(), "ncit/ncit_embeddings2/entity_embeddings.pt"))

if __name__ == "__main__":
    main()