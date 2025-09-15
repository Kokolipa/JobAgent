import json
import os
from collections import Counter
from typing import Any, Dict, List, Union

import langextract as lx
from dotenv import load_dotenv
from google import genai

# Load environment variable and api key 
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


######################################
# <<<< SOFT SKILLS >>>>
######################################

def formulate_softskills_examples(ontology_path: str="../data/langextract/softskills/soft_skills.json") -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    """
    Loads the soft-skills ontology and filters it to a specific size per group, and formats it for LangExtract.

    Args:
        ontology_path (str): The file path to the soft-skills JSON data.

    Returns:
        List[Dict[str, Union[str, List[Dict[str, str]]]]]: The formatted list of examples.
    """

    # Load softskills ontology
    try:
        with open(ontology_path, "r") as file: 
            soft_skill_ontology: List[Dict[str, Any]] = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{ontology_path}' was not found.")
        return []

    # Create an object to store examples
    examples: List[Dict[str, str | List[Dict[str, str]]]] = []
    example_size: int = 3

    # Create an object to limit the amount of examples to be provided to LangExtract 
    group_counts: Dict[str, int] = {}
    for example in soft_skill_ontology: 
        # Extract key elements from softskill ontology 
        extractions = example.get("extractions")
        text = example.get("text")

        # Check if extractions exist
        if extractions and isinstance(extractions, list) and len(extractions) > 0: 
            extraction_class = extractions[0].get("extraction_class")

            # Get the current count for the group, defaulting to 0 if it's the first time
            current_group_count = group_counts.get(extraction_class, 0)
            if current_group_count < example_size: 
                group_counts[extraction_class] = current_group_count + 1
                
                # Formulate LangExtract examples  
                for extraction in extractions:
                    extraction_text = extraction.get("extraction_text")
                    examples.append(
                        lx.data.ExampleData(
                            text=text,
                            extractions=[
                                lx.data.Extraction(
                                    extraction_class=extraction_class,
                                    extraction_text=extraction_text,
                                    attributes={"softskill_type": extraction_class}
                                )
                            ]
                        )
                    )
    
    return examples
    


def extract_softskills_entities(
        resume_dict: Dict[str, str],
        section_keys: List[str],
        output_dir: str = "../src/data/langextract/softskills/"
        ) -> List[Dict[str, str]]: 
    """
    Extracts and categorises soft skills from specified sections of a resume using a
    LangExtract model, and saves the results to a JSONL file.

    This function processes a resume provided as a dictionary, concatenating the text
    from relevant sections. It then uses a pre-defined prompt and examples to
    instruct a large language model (LLM) to extract soft skill entities. The extracted
    entities are categorized into groups like 'communication' or 'leadership'.
    The final results are saved as an annotated JSON Lines file and also returned
    as a list of dictionaries.

    Args:
        resume_dict (Dict[str, str]): 
            A dictionary where keys are resume section names (e.g., 'experience')
            and values are the text content of those sections.
        section_keys (List[str]): 
            A list of keys from `resume_dict` specifying which sections of the
            resume to concatenate and analyze.
        output_dir (str, optional): 
            The directory path where the annotated JSONL file will be saved.
            Defaults to "../src/data/langextract/softskills/".

    Returns:
        List[Dict[str, str]]: 
            A list of dictionaries, where each dictionary represents an extracted
            soft skill entity. Each dictionary contains the `extraction_class`
            (e.g., 'soft_skills') and the `extraction_text` (the extracted phrase).
            The `softskill_group` attribute is part of the saved JSONL but is
            not included in the returned list for a simplified output.
    """

    # Formulate examples: 
    examples = formulate_softskills_examples()

    # Concatenate relivent resume sections
    section_keys = section_keys
    input_text: str = "\n\n".join([resume_dict[key] for key in section_keys])

    # The prompt for LangExtract has to descrive the extraction class 
    EXTRACTION_PROMPT: str = ("""\
    Extract communication, leadership, problem solving, teamwork collaboration, analytical thinking, adaptability, creativity and innovation, and time management optimisation using attributes to group soft skills related information: 
    1. Extract entities in the order they appear in the text.
    2. Use the exact text for extractions. Do not paraphrase or overlap entities.
    3. Soft skill groups can have different values but should always have the same key "softskill_type".
    """)

    # Extract entities
    results = lx.extract(
        text_or_documents=input_text,
        prompt_description=EXTRACTION_PROMPT,
        examples=examples,
        model_id="gemini-2.5-flash",
    )

    # Save results as JSON Line
    lx.io.save_annotated_documents(
        annotated_documents=iter([results]),
        output_dir=output_dir,
        output_name="softskills.jsonl"
    )

    return [
        {
            "extraction_class": result.extraction_class,
            "extraction_text ": result.extraction_text 
        }
        for result in results.extractions
    ]


######################################
# <<<< HARD SKILLS >>>>
######################################

def formulate_skill_examples(json_path: str = "../src/data/langextract/skills/skills.json") -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    """
    Loads the skills.json file and formats it for LangExtract.

    Args:
        json_path (str): The file path to the skills JSON data.

    Returns:
        List[Dict[str, Union[str, List[Dict[str, str]]]]]: The formatted list of examples.
    """

    # Load skills.json 
    skills_path: str = json_path
    with open(skills_path, "r") as file: 
        skills_json = json.load(file)
    
    # Create an object to store examples
    skill_examples = []

    # Iterate and formulate skill_examples
    for example in skills_json: 
        # Extract key elements from skills.json
        text = example.get("text")

        if len(example.get("extractions")) > 1:
            for extraction in  example.get("extractions"):
                extraction_class = extraction.get("extraction_class")
                extraction_text = extraction.get("extraction_text")
                # Formulate LangExtract examples
                skill_examples.append(
                    lx.data.ExampleData(
                        text=text,
                        extractions=[
                            lx.data.Extraction(
                                extraction_class=extraction_class,
                                extraction_text=extraction_text,
                                attributes={"skill_type": extraction_class}
                            )
                        ]
                    )
                )
        else:
            extraction_class = extraction.get("extraction_class")
            extraction_text = extraction.get("extraction_text") 

            # Formulate LangExtract examples
            skill_examples.append(
                lx.data.ExampleData(
                    text=text,
                    extractions=[
                        lx.data.Extraction(
                            extraction_class=extraction_class,
                            extraction_text=extraction_text,
                            attributes={"skill_type": extraction_class}
                        )
                    ]
                )
            )

    return skill_examples


def extract_skill_entities(
        resume_dict: Dict[str, str],
        section_keys: List[str],
        output_dir: str = "../src/data/langextract/softskills/"
        ) -> List[Dict[str, str]]: 
    """
    Extracts and categorises skills from specified sections of a resume using a
    LangExtract model, and saves the results to a JSONL file.

    This function processes a resume provided as a dictionary, concatenating the text
    from relevant sections. It then uses a pre-defined prompt and examples to
    instruct a large language model (LLM) to extract skill entities. The extracted
    entities are categorized into groups like 'cloud_services' or 'databases'.
    The final results are saved as an annotated JSON Lines file and also returned
    as a list of dictionaries.

    Args:
        resume_dict (Dict[str, str]): 
            A dictionary where keys are resume section names (e.g., 'experience')
            and values are the text content of those sections.
        section_keys (List[str]): 
            A list of keys from `resume_dict` specifying which sections of the
            resume to concatenate and analyze.
        output_dir (str, optional): 
            The directory path where the annotated JSONL file will be saved.
            Defaults to "../src/data/langextract/softskills/".

    Returns:
        List[Dict[str, str]]: 
            A list of dictionaries, where each dictionary represents an extracted
            soft skill entity. Each dictionary contains the `extraction_class`
            (e.g., 'skills') and the `extraction_text` (the extracted phrase/sentence).
            The `skill_type` attribute is part of the saved JSONL but is
            not included in the returned list for a simplified output.
    """

    # Formulate examples
    examples = formulate_skill_examples()

    # Concatenate relivent resume sections
    section_keys = section_keys
    input_text: str = "\n\n".join([resume_dict[key] for key in section_keys])

    # The prompt for LangExtract has to descrive the extraction class 
    EXTRACTION_PROMPT_SKILLS: str = ("""\
    Extract cloud services, databases, dev languages, data visualisation, and algorithms using attributes to group skills related information: 
    1. Extract entities in the order they appear in the text.
    2. Use the exact text for extractions. Do not paraphrase or overlap entities.
    3. Skill groups can have different values but should always have the same key "skill_type".
    """)

    # Extract entities
    results = lx.extract(
        text_or_documents=input_text,
        prompt_description=EXTRACTION_PROMPT_SKILLS,
        examples=examples,
        model_id="gemini-2.5-flash",
    )

    # Save results as JSON Line
    lx.io.save_annotated_documents(
        annotated_documents=iter([results]),
        output_dir=output_dir,
        output_name="softskills.jsonl"
    )

    return [
        {
            "extraction_class": result.extraction_class,
            "extraction_text ": result.extraction_text 
        }
        for result in results.extractions
    ]



######################################
# <<<< GENERAL >>>>
######################################
def weight_entities(entities: List[Dict[str, str]]) -> Dict[str, float]: 
    """Calculates the weighted Avg. per entity class to take into account for the overall softskills calculation

    Args:
        entities (List[Dict[str, str]]): A list of dictionaries containing LangExtract results with 'extraction_class' and 'extraction_text'

    Returns:
        Dict[str, float]: Dictionary with entities as keys and weight as values. 
    """
    overall_entities = len(entities)
    entity_type_count = Counter([extraction["extraction_class"] for extraction in entities])
    
    entity_weight_dict: Dict[str, float] = {}
    for entity, entity_count in entity_type_count.items():
        class_weight = entity_count / overall_entities
        weighted_avg = (class_weight * entity_count) / overall_entities
        entity_weight_dict[entity] = weighted_avg
    
    return entity_weight_dict