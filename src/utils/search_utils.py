import asyncio
import re
from typing import Any, Dict, List

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import BaseTool
from tqdm import trange

from prompt_engineering import SEARCH_TOOL_COMPANY_WEBSITE, SEARCH_TOOL_SYS_PROMPT

####################################################
# <<<< SEARCH TOOL UTILS: Sentiment Analysis >>>>
####################################################

N_REVIEWS: int = 10
async def search_reviews(company: str) -> List[Dict[str, Any]]:
    """
    This function will return N_REVIEWS for a single company and add the company name to each result.
    """
    query = f"Provide a review on {company}"
    # Define a search tool 
    search_tool: BaseTool = TavilySearchResults(
        description=SEARCH_TOOL_SYS_PROMPT,
        verbose=True,
        response_format="content_and_artifact",
        max_results=N_REVIEWS,
        search_depth="advanced",
        include_domains=["glassdoor.com.au", "indeed.com"],  
        include_images=False
    )
    # Asnchnously execute the search tool
    outputs = await search_tool.ainvoke(query)
    
    # Add the company name to the resulting output 
    output_with_company = [{**out, "company": company} for out in outputs]
    return sorted(
        output_with_company, reverse=True, key=lambda x: x["score"]
    )

async def search_reviews_all_companies(companies: List[str], batch_size: int = 2) -> List[Dict[str, Any]]:
    """
    Executes a list of companies in batches of 2.
    Returns a flat list of dicts, each with an added 'company' key.
    """
    all_results: List[Dict[str, Any]] = []
    
    for i in trange(0, len(companies), batch_size):
        batch: List[str] = companies[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1}")
        
        # TaskGroup will await the coroutines for us in the event loop 
        async with asyncio.TaskGroup() as tg: 
            # Create and schedule a task for each query in the batch
            tasks = [tg.create_task(search_reviews(company)) for company in batch]
        
        for task in tasks:
            outputs: List[Dict[str, Any]] = task.result()
            all_results.extend(outputs)
    
    return all_results

def filter_and_format_reviews(obj: List[Dict[str, Any]]) -> List[Dict[str, Any]]: 
    """
    Filter a list of review objects to include only those whose 'url' field 
    contains the substring 'review' (case-insensitive).

    :param obj: A list of dictionaries, each expected to have a 'url' key.
    :type obj: List[Dict[str, Any]]
    :return: A filtered list containing only dictionaries where the 'url' includes 'review'.
    :rtype: List[Dict[str, Any]]
    """
    # Filter reviews 
    filtered_reviews: List[Dict[str, Any]] = [
        review for review in obj 
        if re.search("review", review["url"], flags=re.IGNORECASE)
    ]
    # Clean the 'content' field of each filtered review
    for rev in filtered_reviews:
        if "content" in rev and isinstance(rev["content"], str):
            rev["content"] = re.sub(
                pattern=r"[\s\-]+|\[.*?\]|#",
                repl=" ",
                string=rev["content"]
            ).strip()

    return filtered_reviews




####################################################
# <<<< SEARCH TOOL UTILS: Companies overview >>>>
####################################################

async def get_company_overview(company: str) -> Dict[str, Any]: 
    """
    Fetches, cleans, and formats a company's "About Us" page content.
    (This is the helper function you refined in the previous turns)
    """

    QUERY: str = "site: https://{company} about us page"
    search_tool: BaseTool = TavilySearchResults(
        description=SEARCH_TOOL_COMPANY_WEBSITE,
        verbose=True,
        response_format="content_and_artifact",
        max_results=2,
        search_depth="advanced",
        include_domains=["*.com.au", "*.com", ".gov.au"],  
        include_images=False,
        include_raw_content=True,
    )
    results = await search_tool.ainvoke(QUERY.format(company=company))
    
    for result in results:
        
        url: str = result.get("url")
        title: str = result.get("title")
        content: str = result.get("raw_content") or result.get("content")
        if content:
            # Remove content within brackets and parenthese
            content_formatted = re.sub(r"\[.*?\]|\(.*?\)", "", content) 

            # Replace multiple spaces, pipes, exclamation marks, and asterisks with a single space
            content_formatted = re.sub(r"[\s\|\!\*:\)\(,\-—=]+", " ", content_formatted) 

            # Clean up any leading/trailing whitespace
            content_formatted = content_formatted.strip(r":)(]-").strip() 

            overview_formatted: str = "#" + title + ":\n" + content_formatted
            return {
                "url": url,
                "content": overview_formatted,
                "company": company
            }
        
    return {
        "url": None,
        "content": "No relevant 'About Us' content found.",
        "company": company
    }


async def get_all_companies_overview(companies: List[str]) -> List[Dict[str, Any]]: 
    """
    A tool to concurrently extract the "About Us" page overview 
    for a list of companies.
    """
    all_results: List[Dict[str, Any]] = []
    
    async with asyncio.TaskGroup() as tg: 
        tasks = [tg.create_task(get_company_overview(company=company)) for company in companies]
    
    for task in tasks: 
        results: Dict[str, Any] = task.result()
        all_results.append(results)
    return all_results