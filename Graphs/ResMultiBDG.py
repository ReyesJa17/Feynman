from neo4j import GraphDatabase
import json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["URI_NEO4J"] 
os.environ["USER_NEO4J"]
os.environ["PASSWORD_NEO4J"]


# Replace these values with your actual Neo4j cloud connection details
uri = os.environ["URI_NEO4J"]
user = os.environ["USER_NEO4J"]
password = os.environ["PASSWORD_NEO4J"]

# Create the driver and establish a session
driver = GraphDatabase.driver(uri, auth=(user, password))

def test_connection():
    with driver.session() as session:
        greeting = session.run("RETURN 'Connection successful!' AS message")
        for record in greeting:
            print(record["message"])



#LLM setup

llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
        )

#Utility Functions

def get_categories_from_subsubtopics(subsubtopic_name):
    query = """
    MATCH (subsubtopic {name: $subsubtopic_name})-[:HAS_CATEGORY]->(category)
    RETURN DISTINCT category.name AS category_name
    """
    with driver.session() as session:
        result = session.run(query, subsubtopic_name=subsubtopic_name)
        categories = [record["category_name"] for record in result if record["category_name"]]
    return categories



def get_contains_from_category(category_name):
    query = """
    MATCH (category {name: $category_name})-[:CONTAINS]->(contained_node)
    RETURN contained_node.name AS contained_node_name
    """
    with driver.session() as session:
        result = session.run(query, category_name=category_name)
        contains = []

        # Iterate over the results to collect names of the contained nodes
        for record in result:
            contained_node_name = record["contained_node_name"]
            if contained_node_name:
                contains.append(contained_node_name)

    return contains


def get_formulas_from_equality_relationships(node_name):
    """
    Retrieves all formulas associated with the specified node by directly traversing Equality relationships.

    Args:
        node_name (str): The name of the starting node.

    Returns:
        list: A list of unique formulas extracted from the Equality relationships.
    """
    query = """
    MATCH (node:Node {name: $node_name})-[rel:Equality]->(other_node)
    WHERE rel.formula IS NOT NULL
    RETURN DISTINCT rel.formula AS formula
    """
    with driver.session() as session:
        result = session.run(query, node_name=node_name)
        formulas = [record["formula"] for record in result]
    return formulas



def extract_lowest_level_categories_from_file(file_path):
    """
    Loads JSON data from a file and extracts the lowest level category names.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of lowest-level category names.
    """
    try:
        # Open and load the JSON data from the file
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        # Define a list to hold the lowest-level categories
        lowest_level_categories = []

        def recurse_categories(data):
            for category, subcategories in data.items():
                if isinstance(subcategories, dict) and subcategories:
                    # If subcategories exist, go deeper
                    recurse_categories(subcategories)
                else:
                    # No subcategories, so this is a lowest-level category
                    lowest_level_categories.append(category)

        # Start recursion with the loaded JSON data
        recurse_categories(json_data)

        return lowest_level_categories

    except Exception as e:
        return {"error": str(e)}


def remove_quotes(s):
    return s.strip("'")


#Prompt Template

prompt_decide_category = PromptTemplate(
    template= 
    """
    Your task is to decide which category does the next user query belong to. \n
    The query is: '{query}' \n
    The possible categories are: \n
    '{categories}' \n
    Please select the most appropriate category from the list and respond just with the category name.\n
    Always return the choosen category as answer. \n
    
    """,
    input_variables=["query", "categories"]
)


#Chain

chain_decide_category = prompt_decide_category | llm | StrOutputParser()


def get_decision_contain(query):
    # Test the connection



    
    subtopics = extract_lowest_level_categories_from_file("Graphs/termo.json")
    subtopic = chain_decide_category.invoke({"query": query, "categories": subtopics})
    subtopic = remove_quotes(subtopic)
    print(subtopic)
    categories = get_categories_from_subsubtopics(subtopic)
    category = chain_decide_category.invoke({"query": query, "categories": categories})
    print(category)
    contains = get_contains_from_category(category)    
    contain = chain_decide_category.invoke({"query": query, "categories": contains})
    print(contain)
    # Close the driver when done
    return contain,contains


def get_formulas(contain):

    formulas = get_formulas_from_equality_relationships(contain)
    #driver.close(
    return formulas

