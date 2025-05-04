# Helpful smaller functions for the main script.

def pivot_key_columns(df_edits, df_sentences, 
                      quality_mapping={'No Error': 'Good', 
                                       'Error': 'Bad',
                                       'Trivial': 'Neutral'},
                      id_column='Sentence ID', 
                      type_column='Edit Type', 
                      quality_column='Quality',
                      edit_types=None, 
                      qualities=None):
    """
    Pivot edit types and qualities into separate columns with custom quality labels.
    Ensure all edit type and quality combinations exist as columns.
    """
    # Map quality values
    df_edits = df_edits.copy()
    df_edits[quality_column] = df_edits[quality_column].map(quality_mapping)
    
    # Pivot the edit types and qualities
    df_pivot = df_edits.groupby([id_column, type_column, quality_column]).size().unstack(
        fill_value=0
    ).unstack(fill_value=0)

    # Flatten column names
    df_pivot.columns = [f'{quality} ({edit})' for edit, quality in df_pivot.columns]
    
    # Reset index, merge and clean up
    df_pivot = df_pivot.reset_index()
    result_df = df_sentences.merge(df_pivot, on=id_column, how='left')
    result_df = result_df.fillna(0)

    # Ensure all edit type and quality combinations exist
    if edit_types and qualities:
        for edit_type in edit_types:
            for quality in qualities:
                column_name = f"{quality} ({edit_type})"
                if column_name not in result_df.columns:
                    result_df[column_name] = 0  # Add missing column with zeros

    # Capitalize column names
    result_df.columns = result_df.columns.str.replace(r'^([a-z])', lambda x: x.group(1).upper(), regex=True)
    
    return result_df




QUALITY_EXPANDED_MAPPING = {
    # Deletions
    'Good Deletion': 101,
    'Bad Deletion': 102,
    'Trivial Deletion': 103,
    
    # Insertions
    'Good Insertion': 201,
    'Bad Insertion': 202,
    'Trivial Insertion': 203,
    
    # Substitutions
    'Good Substitution': 301,
    'Bad Substitution': 302,
    'Trivial Substitution': 303,
    
    # Reorders
    'Good Reorder': 401,
    'Bad Reorder': 402,
    'Trivial Reorder': 403,
    
    # Splits
    'Good Split': 501,
    'Bad Split': 502,
    'Trivial Split': 503,
    
    # Structure
    'Good Structure': 601,
    'Bad Structure': 602,
    'Trivial Structure': 603,
    
    # Special cases
    'no_edit': 0
}


# turn complex (SALSA, human-generated) format into semicomplex format for 1:1 comparison
# A simple mapping from descriptive significance to an integer value.
SIGNIFICANCE_MAP = {
    "minor": 1,
    "medium": 2,
    "major": 3,
    "trivial": 0
}

def extract_substrings(text, idx_list):
    """Given a text and a list of [start, end] pairs, extract the substrings."""
    return [text[start:end] for start, end in idx_list]

def determine_quality_and_significance(edit):
    """
    Look at the edit's annotation (if present) and decide on quality and significance.
    Uses a simple heuristic based on known keys.
    """
    annotation = edit.get("annotation", {})
    quality = "good"      # default quality
    significance = 0      # default significance

    cat = edit.get("category", "").lower()
    
    if cat == "deletion":
        # Look for a deletion_type annotation
        deletion_info = annotation.get("deletion_type", {})
        if deletion_info:
            val = deletion_info.get("val", "")
            if "bad" in val:
                quality = "bad"
            elif "good" in val:
                quality = "good"
            elif "trivial" in val:
                quality = "trivial"
            # try to get a subvalue (e.g. "minor", "somewhat", "a lot")
            key = val.split("_")[0]  # e.g. from "bad_deletion" get "bad"
            sub_val = deletion_info.get(key, None)
            significance = SIGNIFICANCE_MAP.get(sub_val, 1)
        else:
            quality = "good"
            significance = 0

    elif cat == "substitution":
        sub_info = annotation.get("substitution_info_change", {})
        if sub_info:
            # If there is a "less" field, use that to decide quality.
            if "less" in sub_info:
                sub_val = sub_info["less"].get("val", "")
                if "bad" in sub_val:
                    quality = "bad"
                elif "good" in sub_val:
                    quality = "good"
                elif "trivial" in sub_val:
                    quality = "trivial"
                # If no further detail, default significance:
                significance = 3 if quality == "bad" else 1
            else:
                same_info = sub_info.get("same", {})
                sub_val = same_info.get("val", "")
                if "bad" in sub_val:
                    quality = "bad"
                elif "good" in sub_val:
                    quality = "good"
                elif "trivial" in sub_val:
                    quality = "trivial"
                significance = 3 if quality == "bad" else 1
        else:
            quality = "good"
            significance = 0

    elif cat == "reorder":
        reorder_info = annotation.get("reorder_level", {})
        if reorder_info:
            inner = None
            for key, value in reorder_info.items():
                if isinstance(value, dict) and "val" in value:
                    inner = value.get("val", "")
                    break
            if inner:
                if "bad" in inner:
                    quality = "bad"
                elif "good" in inner:
                    quality = "good"
                elif "trivial" in inner:
                    quality = "trivial"
                significance = SIGNIFICANCE_MAP.get(inner, 1)
            else:
                quality = "good"
                significance = 0
        else:
            quality = "good"
            significance = 0

    elif cat == "insertion":
        insert_info = annotation.get("insertion_type", {})
        if insert_info:
            val = insert_info.get("val", "")
            if "bad" in val:
                quality = "bad"
            elif "good" in val:
                quality = "good"
            elif "trivial" in val:
                quality = "trivial"
            key = val.split("_")[0] if "_" in val else val
            sub_val = insert_info.get(key, None)
            significance = SIGNIFICANCE_MAP.get(sub_val, 1)
        else:
            quality = "good"
            significance = 0

    elif cat == "structure":
        struct_info = annotation.get("structure_type", {})
        if struct_info:
            val = struct_info.get("val", "")
            if "good" in val or val in ("clausal", "pos"):
                quality = "good"
            elif "bad" in val:
                quality = "bad"
            else:
                quality = "trivial"
            # Check for an "impact" value if available
            impact = annotation.get("impact", {})
            if impact:
                for k, v in impact.items():
                    if isinstance(v, str):
                        significance = SIGNIFICANCE_MAP.get(v, 1)
                        break
            else:
                significance = 1
        else:
            quality = "good"
            significance = 0

    else:
        quality = "good"
        significance = 0

    return quality, significance

def convert_complex_to_simple(complex_data):
    """
    Given a JSON-loaded list in the complex format, return a list of examples in the simple format.
    """
    simple_examples = []
    
    # Process each example
    for example in complex_data:
        source = example.get("source", "")
        target = example.get("target", "")
        metadata = example.get("metadata", {})
        thresh_id = example.get("_thresh_id")
        
        simple_example = {
            "source": source,
            "target": target,
            "metadata": metadata,
            "edits": [],
            "_thresh_id": thresh_id
        }
        
        # A recursive function to flatten edits.
        def process_edits(edits, parent_category=None):
            flat_edits = []
            for edit in edits:
                # If there are constituent edits, process them recursively.
                child_category = edit.get("category", parent_category)
                
                # If the current edit has its own indices, process it.
                has_indices = ("input_idx" in edit) or ("output_idx" in edit)
                if has_indices:
                    new_edit = {}
                    # Use the parent_category if provided, otherwise the edit’s own category.
                    new_edit["category"] = parent_category if parent_category else edit.get("category", "")
                    
                    # Extract input text from the source using input_idx if available.
                    if "input_idx" in edit:
                        new_edit["input_text"] = extract_substrings(source, edit["input_idx"])
                    else:
                        new_edit["input_text"] = []
                    
                    # Similarly for output text from the target.
                    if "output_idx" in edit:
                        new_edit["output_text"] = extract_substrings(target, edit["output_idx"])
                    else:
                        new_edit["output_text"] = []
                    
                    # Determine quality and significance from the annotation (if any)
                    q, sig = determine_quality_and_significance(edit)
                    new_edit["quality"] = q
                    new_edit["significance"] = sig
                    
                    flat_edits.append(new_edit)
                
                # if the edit has a "constituent_edits" field, process those recursively.
                if "constituent_edits" in edit:
                    # pass the parent category from the current edit.
                    flat_edits.extend(process_edits(edit["constituent_edits"], parent_category=edit.get("category", parent_category)))
            
            return flat_edits
        
        # Process the top‐level edits.
        simple_example["edits"] = process_edits(example.get("edits", []))
        simple_examples.append(simple_example)
        
    return simple_examples


def df_to_enhanced_latex(df, caption = "A descriptive title for your table", label = "tab:my_table_label"):
    completeString = ""
    completeString += r"""
\begin{table}[ht]
  \centering
    \caption{""" + caption + r"""}
    \label{""" + label + r"""}"""

    completeString += df.to_latex(
        index=False,
        float_format="{:.2f}".format,
    )
    
    completeString += r"""\end{table}"""
    
    print(completeString)



# ---------------------------------
# Helper: Determine which API a given model uses.
# ---------------------------------
def get_api_type(model: str) -> str:
    """
    Determine the API type based on the model string.
    Models containing "deepseek" or "llama" will use Replicate,
    models containing "sonnet" will use Anthropic, and all others will use Azure.
    """
    if "local" in model.lower():
        return "local"
    if "deepseek" in model or "llama" in model.lower():
        return "replicate"
    elif "sonnet" in model.lower():
        return "anthropic"
    else:
        return "azure"
