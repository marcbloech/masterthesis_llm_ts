import copy
import json
import pandas as pd
import numpy as np
import logging as log

import textstat
from easse.fkgl import corpus_fkgl
from evaluate import load

import spacy

_nlp = None  # A module-level variable to hold the loaded spaCy model
def get_nlp():
    """
    Lazy-load the spaCy model if not already loaded.
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm')
    return _nlp



from enum import Enum
from typing import Any, Dict, Optional

########################################################################
# 0) Preliminary definitions and mappings
########################################################################

class Edit(Enum):
    DELETION = 'Deletion'
    INSERTION = 'Insertion'
    SUBSTITUTION = 'Substitution'
    REORDER = 'Reorder'
    STRUCTURE = 'Structure'
    SPLIT = 'Split'

class Information(Enum):
    LESS = 'Generalization'
    SAME = 'Same Information'
    MORE = 'Elaboration'
    DIFFERENT = 'Different Information'

class Error(Enum):
    REPETITION = 'Repetition'
    CONTRADICTION = 'Contradiction'
    HALLUCINATION = 'Hallucination'
    FACTUAL = 'Factual Error'
    IRRELEVANT = 'Irrelevant'
    COREFERENCE = 'Coreference'
    BAD_DELETION = 'Bad Deletion'
    BAD_REORDER = 'Bad Reorder'
    BAD_STRUCTURE = 'Bad Structure'
    BAD_SPLIT = 'Bad Split'
    UNNECESSARY_INSERTION = 'Unnecessary Insertion'
    UNNECESSARY_DELETION = 'Unnecessary Deletion'
    INFORMATION_REWRITE = 'Information Rewrite'
    COMPLEX_WORDING = 'Complex Wording'

class Structure(Enum):
    VOICE = "Voice"
    POS = "Part of Speech"
    TENSE = "Tense"
    GRAMMAR_NUMBER = "Grammatical Number"
    CLAUSAL = "Clausal Structure"
    TRANSITION = "Transition"
    UNKNOWN = "Unknown Structure Change"

class Family(Enum):
    CONTENT = 'Conceptual'
    SYNTAX = 'Syntax'
    LEXICAL = 'Lexical'

class Quality(Enum):
    QUALITY = 'No Error'
    TRIVIAL = 'Trivial'
    ERROR = 'Error'

class ReorderLevel(Enum):
    WORD = 'Word-level'
    COMPONENT = 'Component-level'

content_errors = [
    Error.HALLUCINATION,
    Error.CONTRADICTION,
    Error.REPETITION,
    Error.IRRELEVANT,
    Error.FACTUAL,
    Error.COREFERENCE,
    Error.BAD_DELETION
]

syntax_errors = [
    Error.BAD_REORDER,
    Error.BAD_STRUCTURE,
    Error.BAD_SPLIT
]

lexical_errors = [
    Error.COMPLEX_WORDING,
    Error.INFORMATION_REWRITE,
    Error.UNNECESSARY_INSERTION
]

# Specify metadata for an empty span
empty_span = {
    'span': None,
    'span_length': None
}

# Maps raw annotation outputs to enums or numbers
mapping = {
    'deletion': 1,
    'substitution': 2,
    'insertion': 3,
    'split': 4,
    'reorder': 5,
    'structure': 6,
    'no_edit': -1
}
reverse_mapping = {v: k for k, v in mapping.items()}

# for inter-annotator agreement calculations
EXPANDED_MAPPING = {
    'deletion_more': 1,
    'deletion_less': 2,
    'insertion_more': 3,
    'insertion_less': 4,
    'substitution_more': 5,
    'substitution_less': 6,
    'substitution_same': 7,
    'reorder_word': 8,
    'reorder_component': 9,
    'split_sentence': 10,
    'structure': 11
}

quality_mapping = {
    'minor': 0,
    'somewhat': 1,
    'a lot': 2,
    'very': 2
}

error_mapping = {
    'yes': True,
    'no': False
}

impact_mapping = {
    'negative': Quality.ERROR,
    'no': Quality.TRIVIAL,
    'positive': Quality.QUALITY
}

information_mapping = {
    'less': Information.LESS,
    'same': Information.SAME,
    'more': Information.MORE,
    'different': Information.DIFFERENT
}

error_type_mapping = {
    'repetition': Error.REPETITION,
    'contradiction': Error.CONTRADICTION,
    'hallucination': Error.FACTUAL,
    'factual': Error.FACTUAL,
    'irrelevance': Error.IRRELEVANT,
}

reorder_mapping = {
    'word': ReorderLevel.WORD,
    'component': ReorderLevel.COMPONENT
}

structure_change_type_mapping = {
    "voice": Structure.VOICE,
    "pos": Structure.POS, 
    "tense": Structure.TENSE,
    "grammar_number": Structure.GRAMMAR_NUMBER,  # Add this variant
    "number": Structure.GRAMMAR_NUMBER,         # Keep this too
    "clausal": Structure.CLAUSAL,
    "transition": Structure.TRANSITION
}


## More functions
def process_del_info(raw_annotation):
    
    rating, error_type = None, None
    
    deletion_type = raw_annotation['deletion_type']['val']
    if deletion_type == 'trivial' or deletion_type == 'trivial_deletion':
        edit_quality = Quality.TRIVIAL
    elif deletion_type == 'good_deletion':
        edit_quality = Quality.QUALITY
        rating = quality_mapping.get(raw_annotation['deletion_type'].get('good_deletion', 'somewhat'))
    elif deletion_type == 'bad_deletion':
        edit_quality = Quality.ERROR
        error_type = Error.BAD_DELETION
        rating = quality_mapping.get(raw_annotation['deletion_type'].get('bad_deletion', 'somewhat'))
        
        coreference = error_mapping[raw_annotation['coreference']]
        if coreference:
            error_type = Error.COREFERENCE
    else:
        log.warn(f"DEBUG: processing deletion info for: {raw_annotation}")
        log.warn("Unknown deletion type. Skipping...")
        return None, None, None, None

    grammar_error = error_mapping[raw_annotation['grammar_error']]

    return edit_quality, rating, error_type, grammar_error

def process_add_info(raw_annotation):
    rating, error_type = None, None
    
    annotation_type = raw_annotation['insertion_type']['val']
    if annotation_type == 'elaboration':
        edit_quality = Quality.QUALITY
        rating = quality_mapping[raw_annotation['insertion_type']['elaboration']]
    elif annotation_type == 'trivial_insertion':
        trivial_info = raw_annotation['insertion_type']['trivial_insertion']
        if trivial_info['val'] == 'yes':
            edit_quality = Quality.QUALITY
            rating = quality_mapping.get(trivial_info.get('yes', 'somewhat'))
        else:
            edit_quality = Quality.TRIVIAL
            rating = None
    else:
        edit_quality = Quality.ERROR
        error_type = error_type_mapping.get(annotation_type, Error.FACTUAL)  # Default to FACTUAL if not found
        rating = quality_mapping[raw_annotation['insertion_type'][annotation_type]]

    # Handle missing 'grammar_error' field
    grammar_error = error_mapping[raw_annotation.get('grammar_error', 'no')]
    
    return edit_quality, rating, error_type, grammar_error


def process_same_info(raw_annotation, edit_type):
    structure_change_type = None
    rating, error_type = None, None

    if edit_type == 'substitution':
        return process_substitution(raw_annotation)
    elif edit_type in ['reorder', 'structure', 'split']:
        return process_reorder_structure_split(raw_annotation, edit_type)
    else:
        raise ValueError(f"Unexpected edit_type: {edit_type}")

def process_substitution(raw_annotation):
    structure_change_type = None
    rating, error_type = None, None
    edit_quality = None

    info_change = raw_annotation['substitution_info_change']
    change_type = info_change['val']
    
    if change_type == 'same':
        return process_same_substitution(info_change['same'])
    elif change_type == 'less':
        return process_less_substitution(info_change['less'])
    elif change_type == 'more':
        return process_more_substitution(info_change['more'])
    elif change_type == 'different':
        return process_different_substitution(info_change['different'])
    else:
        raise ValueError(f"Unknown substitution change type: {change_type}")
    
    grammar_error = error_mapping[raw_annotation['grammar_error']]
    return structure_change_type, edit_quality, rating, error_type, grammar_error

def process_same_substitution(same_info):
    structure_change_type = None
    error_type = None
    if same_info['val'] == 'good':
        edit_quality = Quality.QUALITY
        rating = quality_mapping.get(same_info.get('good', 'somewhat'))
    elif same_info['val'] == 'trivial':
        edit_quality = Quality.TRIVIAL
        rating = None
    else:
        edit_quality = Quality.ERROR
        error_type = Error.COMPLEX_WORDING
        rating = None
    return structure_change_type, edit_quality, rating, error_type, None

def process_different_substitution(different_info):
    structure_change_type = None
    edit_quality = Quality.ERROR
    error_type = Error.INFORMATION_REWRITE
    rating = quality_mapping.get(different_info, 'somewhat')
    return structure_change_type, edit_quality, rating, error_type, None

def process_less_substitution(less_info):
    structure_change_type = None
    rating, error_type = None, None

    if less_info['val'] == 'good_deletion':
        edit_quality = Quality.QUALITY
        rating = quality_mapping[less_info['good_deletion']]
    elif less_info['val'] == 'trivial_deletion':
        edit_quality = Quality.TRIVIAL
        rating = None
    else:  # bad_deletion
        edit_quality = Quality.ERROR
        error_type = Error.BAD_DELETION
        rating = quality_mapping.get(less_info.get('bad_deletion', 'somewhat'))
    return None, edit_quality, rating, error_type, None

def process_more_substitution(more_info):
    structure_change_type = None
    rating, error_type = None, None

    if more_info['val'] == 'elaboration':
        edit_quality = Quality.QUALITY
        rating = quality_mapping.get(more_info.get('elaboration', 'somewhat'))
    else:
        edit_quality = Quality.ERROR
        error_type = error_type_mapping.get(more_info['val'], Error.FACTUAL)
        rating = None
    return None, edit_quality, rating, error_type, None

def process_reorder_structure_split(raw_annotation, edit_type):
    structure_change_type = None
    error_type = None

    if edit_type == 'reorder':
        reorder_level = raw_annotation['reorder_level']
        level = reorder_level['val']
        quality_info = reorder_level[level]
    elif edit_type in ['structure', 'split']:
        quality_info = raw_annotation['impact']
        
        # Get structure change type if it's a structure edit
        if edit_type == 'structure':
            structure_change_type = Structure.UNKNOWN
            
            # Check structure_type field exists and has required subfields
            if isinstance(raw_annotation.get('structure_type'), dict):
                structure_type = raw_annotation['structure_type']
                if 'val' in structure_type:
                    structure_val = structure_type['val']
                    log.debug(f"Found structure type value: {structure_val}")
                    structure_change_type = structure_change_type_mapping.get(structure_val)
                    if structure_change_type is None:
                        log.warning(f"Unknown structure type: {structure_val}")
                    else:
                        log.debug(f"Mapped to enum: {structure_change_type}")
                else:
                    log.warning("Structure type missing 'val' field")
            else:
                log.warning(f"Structure edit missing or malformed structure_type field; annotation is: {raw_annotation}")
            
    if quality_info['val'] == 'good':
        edit_quality = Quality.QUALITY
        rating_key = quality_info.get('good', 'somewhat')  
        rating = quality_mapping.get(rating_key, 1)
    elif quality_info['val'] == 'trivial':
        edit_quality = Quality.TRIVIAL
        rating = None
    else:  # bad
        edit_quality = Quality.ERROR
        rating = quality_mapping.get(quality_info.get('bad', 'somewhat'))
        if edit_type == 'reorder':
            error_type = Error.BAD_REORDER
        elif edit_type == 'structure':
            error_type = Error.BAD_STRUCTURE
        elif edit_type == 'split':
            error_type = Error.BAD_SPLIT

    grammar_error = error_mapping[raw_annotation['grammar_error']]

    return structure_change_type, edit_quality, rating, error_type, grammar_error


def process_diff_info(raw_annotation):
    structure_change_type = None
    rating, error_type = None, None
    # ['very', 'no']
    
    rating, grammar_error = raw_annotation
    if grammar_error == '':
        log.debug(f"Couldn't process grammar for substitution: {raw_annotation}. Assuming 'no'...")
        grammar_error = 'no'
    rating, grammar_error = quality_mapping[rating], error_mapping[grammar_error]
    
    return Quality.ERROR, rating, Error.INFORMATION_REWRITE, grammar_error


def calculate_edit_length(original_span, simplified_span):
    orig_len, simp_len = 0, 0
    if original_span is not None:
        for span in original_span:
            orig_len = span[1] - span[0]
    if simplified_span is not None:
        for span in simplified_span:
            simp_len = span[1] - span[0]
    return abs(simp_len - orig_len)

### For edit quality classification:
########################################################################
# Safe getter and edit-quality classifier
########################################################################

def safe_get(dct: Any, *keys, default=None):
    """
    Safely get a nested dictionary value. 
    Example: safe_get(annotation, 'substitution_info_change', 'val')
    """
    for key in keys:
        if not isinstance(dct, dict):
            return default
        dct = dct.get(key, default)
    return dct


def get_edit_quality(annotation: Dict, category: str) -> Optional[str]:
    """
    Returns a string classification of an edit 
    (e.g., "Good Insertion", "Bad Deletion", "Good Split", etc.)
    based on the annotation dictionary and category.
    """
    if category == 'deletion':
        deletion_type = safe_get(annotation, 'deletion_type', 'val')
        if deletion_type == 'good_deletion':
            return 'Good Deletion'
        elif deletion_type == 'bad_deletion':
            return 'Bad Deletion'
        elif deletion_type == 'trivial_deletion':
            return 'Trivial Deletion'
        else:
            return "ERROR COULDNT FIND"

    elif category == 'insertion':
        insertion_type = safe_get(annotation, 'insertion_type', 'val')
        rating_subfield = safe_get(annotation, 'insertion_type', insertion_type)  

        good_insertions = ['elaboration', 'clarification', 'addition']
        bad_insertions = ['repetition', 'contradiction', 'irrelevant', 
            'factual_error', 'facutal_error', 'hallucination']
        if insertion_type in good_insertions:
            return 'Good Insertion'
        elif insertion_type in bad_insertions:
            return 'Bad Insertion'
        else:
            return "Trivial Insertion"

    elif category == 'substitution':
        # Look for the 'substitution_info_change' block
        subst_info = safe_get(annotation, 'substitution_info_change')
        if subst_info:
            val = safe_get(subst_info, 'val')  # e.g. 'same', 'different', 'less', or 'more'
            if val in ['same', 'less', 'more']:
                # For same/less/more, rely on your existing rating logic
                rating = safe_get(subst_info, val, 'val')
                if rating in ['good', 'good_deletion', 'trivial_deletion', 'elaboration']:
                    return 'Good Substitution'
                elif rating in ['bad', 'bad_deletion']:
                    return 'Bad Substitution'
                elif rating == 'elaboration':
                    return 'Good Substitution (Elaboration)'
                elif rating == 'trivial':
                    return 'Trivial Substitution'
                else:
                    # If rating is one of the recognized "error_types", treat as bad
                    error_types = ['repetition', 'contradiction', 'irrelevant',
                                'factual_error', 'facutal_error', 'hallucination']
                    if rating in error_types:
                        return 'Bad Substitution'
                    else:
                        return "ERROR COULDNT FIND"
            elif val == 'different':
                # "different" => Bad Substitution:
                return 'Bad Substitution'
                
                # still read the field
                rating = safe_get(subst_info, 'different')
                return 'Bad Substitution'

            else:
                # Unknown val
                return "ERROR COULDNT FIND"
        else:
            return "ERROR COULDNT FIND"

    elif category == 'reorder':
        reorder_level = safe_get(annotation, 'reorder_level', 'val')
        if reorder_level == 'word_level':
            word_level_info = safe_get(annotation, 'reorder_level', 'word_level')
            rating = safe_get(word_level_info, 'val')
            if rating == 'bad':
                return 'Bad Reorder'
            elif rating in ['good', 'trivial']:
                return 'Good Reorder'
            else:
                return "ERROR COULDNT FIND"
        elif reorder_level == 'component_level':
            component_level_info = safe_get(annotation, 'reorder_level', 'component_level')
            rating = safe_get(component_level_info, 'val')
            if rating == 'bad':
                return 'Bad Reorder'
            elif rating in ['good', 'trivial']:
                return 'Good Reorder'
            else:
                return "ERROR COULDNT FIND"

    elif category == 'split':
        impact = safe_get(annotation, 'impact', 'val')  # e.g. "trivial"
        if impact == 'good':
            return 'Good Split'
        elif impact == 'bad':
            return 'Bad Split'
        elif impact == 'trivial':
            return 'Trivial Split'
        else:
            return "ERROR COULDNT FIND"

    elif category == 'structure':
        impact = safe_get(annotation, 'impact', 'val')  # e.g. "trivial"
        if impact == 'good':
            return 'Good Structure'
        elif impact == 'bad':
            return 'Bad Structure'
        elif impact == 'trivial':
            return 'Trivial Structure'
        else:
            return "ERROR COULDNT FIND"

    # If we cannot determine a "good" or "bad" classification
    return None

# Simplified version of the function above for inter-annotator agreement calculations
def determine_edit_subtype(edit: Dict, category: str) -> int:
    """
    Determines the specific sub-type of an edit based on its category and annotation.
    Returns the corresponding value from EXPANDED_MAPPING.
    """
    if not edit.get('category') or not edit.get('annotation'):
        return -1  # Invalid/unknown
        
    if category == 'deletion':
        deletion_type = edit['annotation'].get('deletion_type', {}).get('val')
        if deletion_type == 'good_deletion':
            return EXPANDED_MAPPING['deletion_less']
        elif deletion_type == 'bad_deletion':
            return EXPANDED_MAPPING['deletion_more']
            
    elif category == 'insertion':
        insertion_type = edit['annotation'].get('insertion_type', {}).get('val')
        if insertion_type == 'elaboration':
            return EXPANDED_MAPPING['insertion_more']
        else:
            return EXPANDED_MAPPING['insertion_less']
            
    elif category == 'substitution':
        subst_info = edit['annotation'].get('substitution_info_change', {})
        val = subst_info.get('val')
        if val == 'same':
            return EXPANDED_MAPPING['substitution_same']
        elif val == 'more':
            return EXPANDED_MAPPING['substitution_more']
        elif val == 'less':
            return EXPANDED_MAPPING['substitution_less']
            
    elif category == 'reorder':
        reorder_level = edit['annotation'].get('reorder_level', {}).get('val')
        if reorder_level == 'word_level':
            return EXPANDED_MAPPING['reorder_word']
        else:
            return EXPANDED_MAPPING['reorder_component']
            
    elif category == 'split':
        return EXPANDED_MAPPING['split_sentence']
        
    elif category == 'structure':
        return EXPANDED_MAPPING['structure']
        
    return -1  # Unknown/invalid type



def get_edit_subtype(edit_type: str, edit_quality: Quality, error_type: Error = None, 
                     reorder_level: ReorderLevel = None, structure_type: Structure = None,
                     information_impact: Information = None) -> str:
    """Maps combination of edit attributes to specific subtype classification"""
    
    # Handle trivial changes first
    if edit_quality == Quality.TRIVIAL:
        return "Trivial Change"
        
    # Handle errors
    if edit_quality == Quality.ERROR:
        if error_type:
            if error_type == Error.BAD_DELETION:
                return "Bad Deletion"
            elif error_type == Error.COREFERENCE:
                return "Coreference"
            elif error_type == Error.REPETITION:
                return "Repetition"
            elif error_type == Error.CONTRADICTION:
                return "Contradiction"
            elif error_type == Error.FACTUAL:
                return "Factual Error"
            elif error_type == Error.IRRELEVANT:
                return "Irrelevant"
            elif error_type == Error.BAD_REORDER:
                return "Bad Word-level Reorder" if reorder_level == ReorderLevel.WORD else "Bad Component Reorder"
            elif error_type == Error.BAD_STRUCTURE:
                return "Bad Structure"
            elif error_type == Error.BAD_SPLIT:
                return "Bad Split"
            elif error_type == Error.COMPLEX_WORDING:
                return "Complex Wrong"
            elif error_type == Error.INFORMATION_REWRITE:
                return "Information Rewrite"
        #return "Grammar" if grammar_error else None

    # Handle successful edits
    if edit_quality == Quality.QUALITY:
        if edit_type == 'insertion' and information_impact == Information.MORE:
            return "Elaboration"
        elif edit_type == 'deletion' and information_impact == Information.LESS:
            return "Generalization"
        elif edit_type == 'reorder':
            return "Word-level Reorder" if reorder_level == ReorderLevel.WORD else "Component-level Reorder"
        elif edit_type == 'split':
            return "Sentence Split"
        elif edit_type == 'structure':
            return "Structure Change"
        elif edit_type == 'substitution' and information_impact == Information.SAME:
            return "Paraphrase"
            
    return None










########################################################################
# 1) SALSA Data Transformation (similar to transform_salsa_data, consolidate_edits, etc.)
########################################################################

def transform_salsa_data(salsa_data):
    """
    Transform raw SALSA JSON into the internal fields you need 
    (such as 'original_spans', 'simplified_spans', 'annotations', 'edits', etc.).
    """
    transformed_data = []
    for entry in salsa_data:
        metadata = entry.get('metadata', {})
        transformed_entry = {
            'id': entry.get('_thresh_id', -1),
            'thresh_id': entry.get('_thresh_id', -1),
            'system': metadata.get('system', 'unknown'),
            'user': metadata.get('annotator', 'unknown'),
            'original': entry['source'],
            'simplified': entry['target'],
            'original_spans': [],
            'simplified_spans': [],
            'annotations': {
                'deletion': {},
                'insertion': {},
                'substitution': {},
                'split': {},
                'reorder': {},
                'structure': {}
            },
            # fill 'edits' directly:
            'edits': []
        }

        for edit in entry.get('edits', []):
            edit_type = edit['category']
            edit_id = edit['id']
            annotation = edit.get('annotation', {})

            if 'input_idx' in edit and edit['input_idx']: # = is not None
                transformed_entry['original_spans'].extend(edit['input_idx'])
            if 'output_idx' in edit and edit['output_idx']: # = is not None
                transformed_entry['simplified_spans'].extend(edit['output_idx'])

            # If there are constituent edits, gather their spans as well
            if 'constituent_edits' in edit and edit['constituent_edits']: # = is not None
                for const_edit in edit['constituent_edits'] and edit['constituent_edits']: # = is not None
                    if 'input_idx' in const_edit:
                        transformed_entry['original_spans'].extend(const_edit['input_idx'])
                    if 'output_idx' in const_edit:
                        transformed_entry['simplified_spans'].extend(const_edit['output_idx'])

            # Add to annotation dict
            transformed_entry['annotations'][edit_type][edit_id] = annotation

            # Create a merged "edit" structure
            transformed_edit = {
                'type': edit_type,
                'id': edit_id - 1, 
                'original_span': edit.get('input_idx'),
                'simplified_span': edit.get('output_idx'),
                'annotation': annotation,
                'has_constituent_edits': 'constituent_edits' in edit,
                'constituent_edit_count': len(edit.get('constituent_edits', [])),
                'constituent_edits': []
            }

            # Attach constituent edits if present
            if 'constituent_edits' in edit and edit['constituent_edits']:
                for const_edit in edit['constituent_edits'] and edit['constituent_edits']:
                    transformed_const_edit = {
                        'type': const_edit['category'],
                        'id': const_edit['id'] - 1,
                        'original_span': const_edit.get('input_idx'),
                        'simplified_span': const_edit.get('output_idx')
                    }
                    transformed_edit['constituent_edits'].append(transformed_const_edit)

            transformed_entry['edits'].append(transformed_edit)

        transformed_data.append(transformed_entry)

    return transformed_data


def consolidate_annotations(data):
    """
    Generates a 'processed_annotations' list for each sentence,
    containing your final classification of each edit 
    (e.g., family=CONTENT/LEXICAL/SYNTAX, error_type=..., etc.)
    """
    out = copy.deepcopy(data)
    idx = 0

    while idx < len(out):
        sent = out[idx]
        processed = []

        for edit in sent['edits']:
            token_length = 0
            # approximate token length from spans
            if edit['original_span'] is not None:
                for span in edit['original_span']:
                    token_length += len(sent['original'][span[0]:span[1]].split(' '))
            if edit['simplified_span'] is not None:
                for span in edit['simplified_span']:
                    token_length += len(sent['simplified'][span[0]:span[1]].split(' '))

            edit['token_length'] = token_length

            # Now run classification logic:
            try:
                processed_anno = process_annotation(edit)
                processed.append(processed_anno)
            except Exception as e:
                log.warning(f"Skipping edit {edit} due to error: {e}")
                pass

        # Add processed_annotations to the sentence
        sent['processed_annotations'] = processed

        for i in range(len(sent['processed_annotations'])):
            sent['processed_annotations'][i]['size'] /= len(sent['original']) if len(sent['original']) else 1

        out[idx] = sent
        idx += 1

    return out

def process_annotation(edit):
    """
    Classifies the edit based on its 'type' and 'annotation' fields.
    Derives attributes such as family, error type, grammar error, 
    rating severity, and so forth. Returns a dictionary of final metadata.
    """
    # Pull relevant fields
    edit_type = edit['type']            # e.g., "insertion", "deletion", "substitution", ...
    raw_annotation = edit['annotation'] # the nested annotation dict
    if raw_annotation == '' or raw_annotation is None:
        raise Exception(f"process_annotation: Missing or empty annotation for edit: {edit}")
    
    # get the edit classification
    edit_classification = get_edit_quality(raw_annotation, edit_type)

    # Default values (some will be overridden below)
    information_impact = Information.SAME.value
    reorder_level = None
    structure_change_type = None
    edit_quality = Quality.QUALITY      # "No Error", effectively
    error_type = None
    grammar_error = False
    rating = None

    # --- 1) Determine base classification (and info impact) by edit_type ---
    if edit_type == 'insertion':
        # Usually indicates adding content => information_impact = MORE
        information_impact = Information.MORE.value
        edit_quality, rating, error_type, grammar_error = process_add_info(raw_annotation)

    elif edit_type == 'deletion':
        # Usually indicates removing content => information_impact = LESS
        information_impact = Information.LESS.value
        edit_quality, rating, error_type, grammar_error = process_del_info(raw_annotation)

    elif edit_type == 'substitution':
        # Substitution can be same, less, more, or different info
        info_change_val = raw_annotation.get('information_change', 'same') 
        information_impact = information_mapping[info_change_val].value
        structure_change_type, eq, rt, err_t, gram_err = process_same_info(raw_annotation, edit_type)
        edit_quality, rating, error_type, grammar_error = eq, rt, err_t, gram_err

    elif edit_type in ['reorder', 'structure', 'split']:
        # Typically these preserve the "amount" of info => keep default = SAME
        structure_change_type, eq, rt, err_t, gram_err = process_same_info(raw_annotation, edit_type)
        edit_quality, rating, error_type, grammar_error = eq, rt, err_t, gram_err

    else:
        raise ValueError(f"Unknown edit type encountered: '{edit_type}'")

    # --- 2) If any error_type is set, the overall quality is 'ERROR' ---
    # (Because we store if there's an error in that edit)
    if error_type is not None:
        edit_quality = Quality.ERROR

    # --- 3) Determine the 'family' of the edit ---
    # FIXED: Check for syntax edits first
    if edit_type in ['reorder', 'structure', 'split']:
        edit_family = Family.SYNTAX
    # Then check content changes
    elif information_impact != Information.SAME and edit_quality != Quality.TRIVIAL:
        edit_family = Family.CONTENT
    # Finally, lexical changes
    else:
        edit_family = Family.LEXICAL

    # Determine structure sub-type for structural edits
    structure_subtype = None
    if edit_type == 'structure' and structure_change_type:
        structure_subtype = structure_change_type.value  # Get the string value from the enum
    

    # --- 4) Calculate the length (in characters) changed (optional) ---
    size = calculate_edit_length(edit['original_span'], edit['simplified_span'])


    # --- X) Get sub-types ---
    edit_subtype = get_edit_subtype(
        edit_type=edit_type,
        edit_quality=edit_quality,
        error_type=error_type,
        reorder_level=reorder_level,
        structure_type=structure_change_type,
        information_impact=information_impact
    )

    # Override edit_subtype with structure_subtype for structural changes
    if edit_type == 'structure' and structure_change_type:
        edit_subtype = structure_change_type.value


    # gather the constituent edits' indices:
    const_input = []
    const_output = []
    if edit.get('has_constituent_edits'):
        for cedit in edit['constituent_edits']:
            const_input.append(cedit.get('original_span', []))
            const_output.append(cedit.get('simplified_span', []))

    # --- 5) Build final dictionary ---
    return {
        'edit_type': edit_type,                    
        'id': edit['id'],
        'information_impact': information_impact,
        'type': edit_quality,                       # e.g. Quality.ERROR / Quality.QUALITY / Quality.TRIVIAL
        'edit_subtype': edit_subtype,
        'edit_classification': edit_classification if edit_classification else edit_type,
         'family': edit_family,              
        'grammar_error': grammar_error,             
        'error_type': error_type,                   
        'rating': rating,
        'size': size,                               
        'token_size': edit.get('token_length', 0),
        'reorder_level': reorder_level,
        'structure_subtype': structure_subtype,

        'original_span': edit.get('original_span'),
        'simplified_span': edit.get('simplified_span'),
        'has_constituent_edits': edit.get('has_constituent_edits', False),
        'constituent_edit_count': edit.get('constituent_edit_count', 0),

        # New fields:
        'constituent_input_idx': const_input,
        'constituent_output_idx': const_output,
    }


########################################################################
# 2) Convert the processed data into df_edits and df_sentences
########################################################################

def get_edits_per_sentence(data):
    """
    Given the SALSA data with 'processed_annotations', produce 
    a row per *edit* (and sub-edits if you prefer).
    """
    rows = []

    for item in data:
        sentence_id = f"{item['system']}_{item['id']:03d}"

        # If no edits, create a "dummy" row so the sentence is not lost
        if not item.get('processed_annotations', []):
            rows.append({
                'Sentence ID': sentence_id,
                'Source': item['original'],
                'Target': item['simplified'],
                'System': item['system'],
                'Dataset': item.get('dataset', ''),
                'Family': None,
                'Edit Type': 'no_edit',
                'Edit Sub-Type': None,
                'Edit Classification': None,
                'Quality': None,
                'Information Impact': None,
                'Grammar Error': 'No',
                'Significance': 0,
                'Input Index': None,
                'Output Index': None,
                'Constituent Edits': 0,
                'Is Constituent Edit': False,
                'Parent Edit Type': 'N/A',
                'Constituent Input Index': None,
                'Constituent Output Index': None,
            })
            continue

        for ann_edit in item['processed_annotations']:
            # Basic row:
            row_dict = {
                'Sentence ID': sentence_id,
                'Source': item['original'],
                'Target': item['simplified'],
                'System': item['system'],
                'Dataset': item.get('dataset', ''),

                'Family': ann_edit['family'],
                'Edit Type': ann_edit['edit_type'],

                'Edit Sub-Type': ann_edit['edit_subtype'], 
                'Edit Classification': ann_edit.get('edit_classification', None),
                'Structure Sub-Type': ann_edit.get('structure_subtype', None),

                'Quality': ann_edit['type'],
                'Information Impact': ann_edit['information_impact'],
                'Grammar Error': 'Yes' if ann_edit['grammar_error'] else 'No',
                'Significance': ann_edit['rating'] if ann_edit['rating'] is not None else 0,
                'Input Index': ann_edit['original_span'],
                'Output Index': ann_edit['simplified_span'],
                'Constituent Edits': ann_edit['constituent_edit_count'],
                'Is Constituent Edit': False,
                'Parent Edit Type': 'N/A',

                'Constituent Input Index': ann_edit['constituent_input_idx'],
                'Constituent Output Index': ann_edit['constituent_output_idx'],
            }
            rows.append(row_dict)

    return pd.DataFrame(rows)


def create_sentence_level_df(df_edits):
    """
    Roll up (group) the per-edit DataFrame to one row per sentence.
    Summarize or count edit types, etc.
    """
    grouped = df_edits.groupby('Sentence ID')

    # Basic structure
    df_sentence = pd.DataFrame({
        'Sentence ID': grouped.groups.keys(),
        'Source': grouped['Source'].first(),
        'Target': grouped['Target'].first(),
        'System': grouped['System'].first(),
        'Dataset': grouped['Dataset'].first(),
        # Example: total edits, ignoring no_edit
        'Total Edits': grouped.apply(
            lambda x: sum(x['Edit Type'] != 'no_edit')
        )
    }).reset_index(drop=True)

    # Count how many times each Edit Type appears
    edit_types = df_edits['Edit Type'].unique()
    for etype in edit_types:
        col_name = f"{etype}_count"
        df_sentence[col_name] = grouped.apply(
            lambda x: (x['Edit Type'] == etype).sum()
        ).values

    return df_sentence


########################################################################
# 3) Compute automated metrics (FKGL, ARI, BERTScore, etc.)
########################################################################

# Initialize spacy once

#nlp = spacy.load('en_core_web_sm')

# 1) FKGL
def calculate_fkgl(text):
    """
    Calculates the Flesch-Kincaid Grade Level for a single text 
    using the EASSE library function corpus_fkgl.
    """
    return corpus_fkgl([text])

# 3) Lexical Diversity
def lexical_diversity(text):
    """
    Returns the Type-Token Ratio (TTR) = unique_tokens / total_tokens.
    """
    words = text.split()
    total_tokens = len(words)
    unique_tokens = len(set(words))
    
    return (unique_tokens / total_tokens) if total_tokens > 0 else 0

# 4) Syntactic Complexity
def calculate_syntactic_complexity(sentence):
    """
    Heuristic approach: count clause-level verbs 
    (VERB tokens with certain dependencies) 
    as a measure of complexity.
    """
    # Grab the nlp from get_nlp()
    nlp = get_nlp()
    doc = nlp(sentence)
    clause_heads = set()

    for token in doc:
        # Check if the token is a verb (excluding auxiliaries)
        if token.pos_ == "VERB" and token.dep_ != "aux":
            # If the verb is the root or has a clausal dependency label, it's a clause head
            if token.dep_ in {"ROOT", "csubj", "ccomp", "xcomp", "advcl", "acl", "relcl"}:
                clause_heads.add(token)
            elif token.dep_ == "conj" and token.head.pos_ == "VERB":
                clause_heads.add(token)

    return len(clause_heads)

# 5) GLEU (Google BLEU) from evaluate
google_bleu = load("google_bleu")
def calculate_gleu(text_compl, text_simpl):
    """
    Calculate GLEU between two strings using Hugging Face's 'evaluate' library.
    """
    result = google_bleu.compute(
        predictions=[text_simpl], 
        references=[[text_compl]]
    )
    return result['google_bleu']

# 6) BERTScore
bertscore = load("bertscore")
def calculate_bertscore(text_compl, text_simpl, language="en"):
    """
    Calculates BERTScore (Precision, Recall, F1) between two texts 
    using the HuggingFace evaluate library.
    """
    try:
        results = bertscore.compute(
            predictions=[text_simpl],  # simplified text
            references=[text_compl],   # source text
            lang=language
        )
        return {
            'precision': results['precision'][0],
            'recall':    results['recall'][0],
            'f1':        results['f1'][0]
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {str(e)}")
        return {
            'precision': None,
            'recall': None,
            'f1': None
        }

# 7) LENS-SALSA
from lens import download_model, LENS_SALSA
lens_salsa_path = download_model("davidheineman/lens-salsa")
lens_salsa = LENS_SALSA(lens_salsa_path)

def calculate_LENS(text_compl, text_simpl):
    """
    Returns the LENS-SALSA score for source vs. simplified text.
    """
    scores, word_level_scores = lens_salsa.score(
        [text_compl], 
        [text_simpl]
    )
    return scores[0]

##############################################################################
# Final aggregator function
##############################################################################

def calculate_all_metrics(row):
    """
    Combine all metric calculations for a single row. 
    Expects the row to have 'Source' and 'Target' columns.

    For each row, the function will add columns:
      - FKGL_Source, FKGL_Target, FKGL_Difference
      - ARI_Source, ARI_Target, ARI_Difference
      - Lexical_Diversity_Source, Lexical_Diversity_Target, Lexical_Diversity_Difference
      - Syntactic_Complexity_Source, Syntactic_Complexity_Target, ...
      - BERTScore_Precision, BERTScore_Recall, BERTScore_F1
      - GLEU
      - LENS_SALSA
    """
    source = row['Source']
    target = row['Target']

    # 1) FKGL
    row['FKGL_Source'] = calculate_fkgl(source)
    row['FKGL_Target'] = calculate_fkgl(target)
    row['FKGL_Difference'] = row['FKGL_Target'] - row['FKGL_Source']

    # 2) ARI
    row['ARI_Source'] = textstat.automated_readability_index(source)
    row['ARI_Target'] = textstat.automated_readability_index(target)
    row['ARI_Difference'] = row['ARI_Target'] - row['ARI_Source']

    # 3) Lexical Diversity
    row['Lexical_Diversity_Source'] = lexical_diversity(source)
    row['Lexical_Diversity_Target'] = lexical_diversity(target)
    row['Lexical_Diversity_Difference'] = (
        row['Lexical_Diversity_Target'] - row['Lexical_Diversity_Source']
    )

    # 4) Syntactic Complexity
    row['Syntactic_Complexity_Source'] = calculate_syntactic_complexity(source)
    row['Syntactic_Complexity_Target'] = calculate_syntactic_complexity(target)
    row['Syntactic_Complexity_Difference'] = (
        row['Syntactic_Complexity_Target'] - row['Syntactic_Complexity_Source']
    )

    # 5) BERTScore
    # split model by language
    if 'wikiDE' in row['Dataset']:
        bertscore_results = calculate_bertscore(source, target, language="de")
        print("Debug: calculating german BERTSCORE")
    else:
        bertscore_results = calculate_bertscore(source, target, language="en")
    row['BERTScore_Precision'] = bertscore_results['precision']
    row['BERTScore_Recall'] = bertscore_results['recall']
    row['BERTScore_F1'] = bertscore_results['f1']

    # 6) GLEU
    row['GLEU'] = calculate_gleu(source, target)

    # 7) LENS-SALSA
    row['LENS_SALSA'] = calculate_LENS(source, target)

    return row


########################################################################
# 4) High-Level Function to unify everything
########################################################################

def unify_salsa_processing(
    file_path_or_data,
    dataset_name=None,
    system_name=None,
    compute_metrics=True
):
    """
    1) Load the SALSA JSON (or accept a list/dict already loaded),
    2) transform it, 
    3) produce 'processed_annotations',
    4) build df_edits and df_sentences,
    5) optionally compute readability metrics, etc.
    """
    # Step A: read JSON if path is given
    if isinstance(file_path_or_data, str):
        with open(file_path_or_data, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        raw_data = file_path_or_data

    # Step B: transform the SALSA data structure
    data = transform_salsa_data(raw_data)

    # Attach dataset/system if needed
    for sent in data:
        if dataset_name:
            sent['dataset'] = dataset_name
        if system_name:
            sent['system'] = system_name

    # Step C: consolidate & produce 'processed_annotations'
    data = consolidate_annotations(data)

    # Step D: build df_edits
    df_edits = get_edits_per_sentence(data)

    # Step E: build df_sentences
    df_sentences = create_sentence_level_df(df_edits)

    # (Optional) compute metrics:
    if compute_metrics:
        df_sentences = df_sentences.apply(calculate_all_metrics, axis=1)

    return df_sentences, df_edits