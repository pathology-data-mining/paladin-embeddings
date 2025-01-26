from owlready2 import *
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union
import logging
from pykeen.triples import TriplesFactory
import torch
import pickle


class NCItoPyKEEN:
    def __init__(self, owl_file_path: str):
        """Initialize the NCIExtractor with path to NCI Thesaurus OWL file."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.onto = get_ontology(owl_file_path).load()
        self.logger.info(f"Loaded ontology: {self.onto.base_iri}")
        
        # Store extracted relations
        self.taxonomic_relations: List[Tuple[str, str, str]] = []
        self.role_relations: List[Tuple[str, str, str]] = []
        self.property_relations: List[Tuple[str, str, str]] = []
        self.restriction_relations: List[Tuple[str, str, str]] = []
        
        # Store relation DataFrames
        self.relations: Dict[str, pd.DataFrame] = {}

    def _validate_entity_name(self, entity) -> str:
        """Validate and clean entity names."""
        if entity is None:
            return None
        if not hasattr(entity, 'name'):
            return str(entity)
        return entity.name.strip()

    def _extract_taxonomic_relations(self):
        """Extract taxonomic (is-a) and equivalent class relations."""
        self.logger.info("Extracting taxonomic relations...")
        
        for cls in self.onto.classes():
            # Handle is-a relations
            for parent in cls.is_a:
                if isinstance(parent, ThingClass):
                    self.taxonomic_relations.append((
                        self._validate_entity_name(cls),
                        'rdfs:subClassOf',
                        self._validate_entity_name(parent)
                    ))
            
            # Handle equivalent classes
            if hasattr(cls, 'equivalent_to'):
                for equiv in cls.equivalent_to:
                    if isinstance(equiv, ThingClass):
                        self.taxonomic_relations.append((
                            self._validate_entity_name(cls),
                            'owl:equivalentClass',
                            self._validate_entity_name(equiv)
                        ))
    
    def _extract_role_relations(self):
        """Extract role relations (object properties)."""
        self.logger.info("Extracting role relations...")
        
        for prop in self.onto.object_properties():
            try:
                # Get direct role relations
                if prop.domain and prop.range:
                    for domain in prop.domain:
                        for range_ in prop.range:
                            if isinstance(domain, ThingClass) and isinstance(range_, ThingClass):
                                self.role_relations.append((
                                    self._validate_entity_name(domain),
                                    prop.name,
                                    self._validate_entity_name(range_)
                                ))
            except Exception as e:
                self.logger.warning(f"Error processing property {prop.name}: {str(e)}")
            
            # Extract actual usage of properties in individuals
            for subject in self.onto.individuals():
                try:
                    objects = getattr(subject, prop.name)
                    if not isinstance(objects, (list, set)):
                        objects = [objects]
                    for obj in objects:
                        if obj is not None:
                            self.role_relations.append((
                                self._validate_entity_name(subject),
                                prop.name,
                                self._validate_entity_name(obj)
                            ))
                except AttributeError:
                    continue
    
    def _extract_property_relations(self):
        """Extract data property relations."""
        self.logger.info("Extracting property relations...")
        
        for prop in self.onto.data_properties():
            if prop.domain:
                for domain in prop.domain:
                    if isinstance(domain, ThingClass):
                        # Handle multiple ranges
                        ranges = prop.range if prop.range else ["xsd:string"]
                        for range_type in ranges:
                            range_name = range_type.name if hasattr(range_type, 'name') else str(range_type)
                            self.property_relations.append((
                                self._validate_entity_name(domain),
                                prop.name,
                                range_name
                            ))
    
    def _extract_restriction_relations(self):
        """Extract relations from OWL restrictions."""
        self.logger.info("Extracting restriction relations...")
        
        for cls in self.onto.classes():
            for restriction in cls.is_a:
                if isinstance(restriction, Restriction):
                    # Handle anonymous classes in restrictions
                    if hasattr(restriction, 'Classes') and restriction.Classes:
                        for target_cls in restriction.Classes:
                            self.restriction_relations.append((
                                self._validate_entity_name(cls),
                                f"restriction_{restriction.property.name}",
                                self._validate_entity_name(target_cls)
                            ))
                    # Existing code for direct restrictions
                    elif restriction.property and restriction.value:
                        if isinstance(restriction.value, ThingClass):
                            self.restriction_relations.append((
                                self._validate_entity_name(cls),
                                f"restriction_{restriction.property.name}",
                                self._validate_entity_name(restriction.value)
                            ))
                        elif isinstance(restriction.value, (int, float, str)):
                            self.restriction_relations.append((
                                self._validate_entity_name(cls),
                                f"restriction_{restriction.property.name}",
                                str(restriction.value)
                            ))

    def _log_statistics(self, relations: Dict[str, pd.DataFrame]):
        """Log statistics about extracted relations."""
        for rel_type, df in relations.items():
            self.logger.info(f"Extracted {len(df)} {rel_type} relations")
            if not df.empty:
                self.logger.info(f"Sample of {rel_type} relations:")
                self.logger.info(df.head())
                
                # Count unique relations
                unique_relations = df['relation'].unique()
                self.logger.info(f"Unique {rel_type} relation types: {len(unique_relations)}")
                self.logger.info(f"Top relation types: {', '.join(unique_relations[:5])}")

    def extract_all_relations(self) -> Dict[str, pd.DataFrame]:
        """Extract all types of relations from the NCI Thesaurus."""
        self._extract_taxonomic_relations()
        self._extract_role_relations()
        self._extract_property_relations()
        self._extract_restriction_relations()
        
        # Convert to DataFrames
        self.relations = {
            'taxonomic': pd.DataFrame(self.taxonomic_relations, 
                                    columns=['subject', 'relation', 'object']),
            'role': pd.DataFrame(self.role_relations, 
                               columns=['subject', 'relation', 'object']),
            'property': pd.DataFrame(self.property_relations, 
                                   columns=['subject', 'relation', 'object']),
            'restriction': pd.DataFrame(self.restriction_relations, 
                                      columns=['subject', 'relation', 'object'])
        }
        
        self._log_statistics(self.relations)
        return self.relations
    
    def create_pykeen_dataset(self, 
                             relation_types: Union[List[str], str] = 'all',
                             relation_names: Union[List[str], None] = None,
                             train_size: float = 0.8,
                             validation_size: float = 0.1) -> Dict[str, TriplesFactory]:
        """
        Convert specified relations to PyKEEN format and create train/validation/test splits.
        
        Args:
            relation_types: List of relation types to include ('taxonomic', 'role', 'property', 'restriction')
                          or 'all' to include all relations
            relation_names: Optional list of specific relation names to include
            train_size: Proportion of data for training
            validation_size: Proportion of data for validation
        
        Returns:
            Dictionary containing PyKEEN TriplesFactory objects for train/validation/test sets
        """
        if not self.relations:
            self.extract_all_relations()
            
        # Determine which relation types to include
        if relation_types == 'all':
            relation_types = list(self.relations.keys())
        elif isinstance(relation_types, str):
            relation_types = [relation_types]
            
        # Combine specified relation types
        combined_relations = pd.concat([
            self.relations[rel_type] for rel_type in relation_types
        ], ignore_index=True)
        
        # Filter by specific relation names if provided
        if relation_names is not None:
            combined_relations = combined_relations[
                combined_relations['relation'].isin(relation_names)
            ]
        
        self.logger.info(f"Creating PyKEEN dataset with {len(combined_relations)} triples")
        
        # Create PyKEEN TriplesFactory
        tf = TriplesFactory.from_labeled_triples(
            combined_relations.values,
            create_inverse_triples=True  # Creates inverse relations automatically
        )
        
        # Calculate split sizes
        test_size = 1.0 - train_size - validation_size
        
        # Split the dataset
        if validation_size > 0:
            training, validation, testing = tf.split([
                train_size,
                validation_size,
                test_size
            ])
            splits = {
                'train': training,
                'validation': validation,
                'test': testing
            }
        else:
            training, testing = tf.split([train_size, test_size])
            splits = {
                'train': training,
                'test': testing
            }
            
        # Log split statistics
        for split_name, split_tf in splits.items():
            self.logger.info(f"{split_name} set: {len(split_tf.mapped_triples)} triples")
            
        return splits
    
    def filter_relations_by_frequency(self,
                                    min_frequency: int = 10,
                                    relation_types: Union[List[str], str] = 'all') -> pd.DataFrame:
        """Filter relations based on minimum frequency."""
        if relation_types == 'all':
            relation_types = list(self.relations.keys())
        elif isinstance(relation_types, str):
            relation_types = [relation_types]
            
        combined_relations = pd.concat([
            self.relations[rel_type] for rel_type in relation_types
        ], ignore_index=True)
        
        # Count relation frequencies
        relation_counts = combined_relations['relation'].value_counts()
        frequent_relations = relation_counts[relation_counts >= min_frequency].index
        
        filtered_relations = combined_relations[
            combined_relations['relation'].isin(frequent_relations)
        ]
        
        self.logger.info(f"Filtered from {len(combined_relations)} to {len(filtered_relations)} triples")
        self.logger.info(f"Retained {len(frequent_relations)} relation types")
        
        return filtered_relations
    
    def create_focused_subset(self,
                            focus_entities: List[str],
                            max_hops: int = 2,
                            relation_types: Union[List[str], str] = 'all') -> pd.DataFrame:
        """Create a subset of relations centered around specific entities."""
        if relation_types == 'all':
            relation_types = list(self.relations.keys())
        elif isinstance(relation_types, str):
            relation_types = [relation_types]
            
        combined_relations = pd.concat([
            self.relations[rel_type] for rel_type in relation_types
        ], ignore_index=True)
        
        entities_to_include = set(focus_entities)
        current_entities = set(focus_entities)
        
        # Expand by hops
        for _ in range(max_hops):
            new_entities = set()
            
            # Forward direction
            related = combined_relations[
                combined_relations['subject'].isin(current_entities)
            ]['object'].unique()
            new_entities.update(related)
            
            # Backward direction
            related = combined_relations[
                combined_relations['object'].isin(current_entities)
            ]['subject'].unique()
            new_entities.update(related)
            
            current_entities = new_entities - entities_to_include
            entities_to_include.update(new_entities)
            
            if not current_entities:
                break
        
        # Filter relations to only include the focused subset
        focused_relations = combined_relations[
            (combined_relations['subject'].isin(entities_to_include)) &
            (combined_relations['object'].isin(entities_to_include))
        ]
        
        self.logger.info(f"Created focused subset with {len(focused_relations)} triples")
        self.logger.info(f"Includes {len(entities_to_include)} entities")
        
        return focused_relations

# Example usage:
"""
# Initialize converter
converter = NCItoPyKEEN('path_to_nci_thesaurus.owl')

# Extract all relations
relations = converter.extract_all_relations()

# Create PyKEEN dataset with specific relation types
pykeen_data = converter.create_pykeen_dataset(
    relation_types=['role', 'taxonomic'],
    relation_names=['has_target', 'has_mechanism_of_action', 'rdfs:subClassOf']
)

# Train a PyKEEN model
from pykeen.pipeline import pipeline

result = pipeline(
    model='RotatE',
    training=pykeen_data['train'],
    testing=pykeen_data['test'],
    validation=pykeen_data['validation'],
    model_kwargs={'embedding_dim': 128},
    training_kwargs={'num_epochs': 100, 'batch_size': 128}
)
"""

if __name__ == "__main__":
    # Initialize converter
    converter = NCItoPyKEEN('ncit/ThesaurusInferred.owl')

    # Extract all relations
    relations = converter.extract_all_relations()

    # Create PyKEEN dataset with specific relation types and names
    pykeen_data = converter.create_pykeen_dataset(
        # relation_types=['role', 'taxonomic'],
        # relation_names=['has_target', 'has_mechanism_of_action', 'rdfs:subClassOf'],
        train_size=0.8,
        validation_size=0.1
    )

    print(pykeen_data)

    with open('ncit/ncit_embeddings/pykeen_data.pkl', 'wb') as f:
        pickle.dump(pykeen_data, f)

    # # Create focused subset around specific drugs
    # drug_subset = converter.create_focused_subset(
    #     focus_entities=['Aspirin', 'Ibuprofen'],
    #     max_hops=2,
    #     relation_types=['role']
    # )

    # # Filter relations by frequency
    # frequent_relations = converter.filter_relations_by_frequency(
    #     min_frequency=10,
    #     relation_types=['role']
    # )

    # # Use the PyKEEN dataset for training
    # training_factory = pykeen_data['train']
    # validation_factory = pykeen_data['validation']
    # testing_factory = pykeen_data['test']