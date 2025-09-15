
import re
from datetime import datetime
from typing import Dict, List, Union

from langchain.document_loaders import PyPDFLoader


def load_resume_pages(path: str) -> List[str]: 
    """
    Loads text content from each page of a PDF document.

    This function uses PyPDFLoader to read a PDF file from the specified path.
    It processes the document in 'page' mode with 'layout' extraction, ensuring 
    that each page's text content is returned as a separate string, preserving
    the original document's layout.

    Args:
        path (str): The file path to the PDF document to be loaded.

    Returns:
        List[str]: A list of strings, where each string represents the full
                   text content of a single page from the PDF.
                   Returns an empty list if the file cannot be loaded.
    
    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        
    Note:
        This function assumes that `PyPDFLoader` is available and correctly
        configured in the environment.
    """

    resume_path: str = path
    loader = PyPDFLoader(
        file_path=resume_path,
        extract_images=False,
        mode="page",
        extraction_mode="layout"
    )
    resume = loader.load()
    resume_pages: List[str] = [page.page_content for page in resume]
    return resume_pages


def parse_resume(doc_pages: List[str], resume_sections: List[str], experience_key: str) -> Dict[str, str]:
    """
    Parses a resume from a list of text pages and extracts content into a dictionary
    based on predefined section headers.

    This function iterates through each page of a resume (provided as a list of strings)
    and uses regular expressions to identify the start and end of specific sections
    (e.g., 'Education', 'Skills'). It handles multi-page resumes by concatenating content
    from continuation sections. The primary goal is to structure the flat text of a resume
    into a more accessible dictionary format, making it easier to analyze specific sections.

    The function assumes that section headers are unique and follow a consistent format
    throughout the document. It also includes specific logic to handle the continuation
    of the 'Professional Experience' section onto subsequent pages.

    Args:
        doc_pages (List[str]):
            A list of strings, where each string is the text content of a single
            page of the resume.
        resume_sections (List[str]):
            A list of strings representing the section headers to be extracted
            from the resume (e.g., ['Education', 'Skills', 'Projects']).
        experience_key (str):
            The specific key name for the professional experience section
            (e.g., 'Professional Experience'). This is used to handle multi-page
            concatenation for this specific section.

    Returns:
        Dict[str, str]:
            A dictionary where keys are the section headers and values are the
            concatenated text content of each corresponding section.

    Note:
        The function's logic is heavily dependent on the specific formatting and
        layout of the resume. Small variations in headers, spacing, or the
        presence of different patterns (like bullet points) may cause it to
        fail or produce incorrect results.
    """
         
    # Define regex patterns: 
    job_experience_pattern: str = r"^(?!•)(.+?\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*[-–]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{0,4})$"
    # Define a list with section names
    sections: List[str] = resume_sections
    resume_dict: Dict[str, str] = {}

    for i, page in enumerate(doc_pages, 1):
        
        # Extract min and max spans: Reset the spans for each page iterated
        max_spans: List[int] = [] 
        min_spans: List[int] = []
        max_span: int = 0
        for section in sections: 
            results = re.finditer(pattern=section, string=page, flags=re.DOTALL)
            for result in results:
                max_spans.append(result.span()[1]) # end span
                max_span = result.span()[1] 
                min_spans.append(result.span()[0])
        
        # Get the size of the sections in the page
        min_span = min(min_spans)
        sections_size = len(max_spans)
        for idx, span in enumerate(max_spans):

            # Firt page logic
            if i == 1: 
                # Extract sections and clean up results
                if span != max_span:
                    section_result = page[max_spans[idx]:max_spans[idx+1]].strip("\n\n").strip()
                    resume_dict[sections[idx]] = section_result
                else:  
                    section_result = page[span:].strip("\n\n").strip()
                    resume_dict[sections[idx]] = section_result
            
            # Page 2+ logic
            else: 
                if span != max_span:
                    section_result = page[max_spans[idx]:max_spans[idx+1]].strip("\n\n").strip()
                    
                    # section_result = re.sub(pattern=dot_points_pattern, repl=" ", string=section_result, flags=re.MULTILINE).strip("\n\n").strip()
                    resume_dict[sections[idx + sections_size]] = section_result
                else:
                    # Extract additional job experience from the second page 
                    job_exper = [match for match in re.finditer(pattern=job_experience_pattern, string=page, flags=re.MULTILINE)]
                    additional_experience = job_exper[0].span()[0]

                    # Additional professional experiences can appear on page 2+ 
                    resume_dict[experience_key] = resume_dict.get(experience_key, "Professional Experience") + "\n" + page[additional_experience:min_span]
                    
                    section_result = page[span:].strip("\n\n").strip()
                    resume_dict[sections[idx + sections_size]] = section_result
    
    return resume_dict


def get_years_of_experience(resume_dict: Dict[str, str], experience_key: str)-> Union[float, None]: 
    """
    Calculates the total years of professional experience from a resume section.

    This function uses a robust regular expression to find and extract all
    month-year date combinations within the specified professional experience section
    of a resume. It then parses these dates using the `datetime` module and
    calculates the total duration in years by finding the difference between
    the earliest and latest dates. The result is rounded to two decimal places.

    Args:
        resume_dict (Dict[str, str]):
            A dictionary containing resume sections, where keys are section names
            (e.g., 'Professional Experience') and values are the text content.
        experience_key (str):
            The key in `resume_dict` that corresponds to the professional
            experience section (e.g., 'Professional Experience').

    Returns:
        Union[float, None]:
            The total years of experience as a floating-point number, rounded to
            two decimal places. Returns `None` if the experience key is not found
            or if no valid dates can be extracted from the text.
    
    Raises:
        ValueError: If the date format in the resume does not match the
                    expected `"%b %Y"` format.
    
    Note:
        This function assumes that the provided experience text contains a
        chronological list of jobs, where the difference between the earliest
        and latest date accurately represents total experience. It does not
        account for gaps in employment.
    """
    experiences = resume_dict.get(experience_key)
    if not experiences:
        return None
    
    # Define regex pattern to locate month-year date formats
    years_pattern = r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-\s/]*(?:19|20)\d{2}\b"
    year_month_format = "%b %Y"

    # Find all matching date strings in the experience text
    year_months = [match for match in re.findall(pattern=years_pattern, string=experiences, flags=re.IGNORECASE)]
    if not year_months:
        return None
    try:
        parse_dates = [datetime.strptime(year, year_month_format) for year in year_months]
    except ValueError:
         # If any date fails to parse, something is wrong with the format. 
        return None 
    
    # Calculate the difference between the most recent and earliest date
    total_days = (max(parse_dates) - min(parse_dates)).days
    if total_days < 0: 
        return None # Handling cases where total_days are illogical
    years_of_experience = total_days / 365

    return round(years_of_experience, 2)