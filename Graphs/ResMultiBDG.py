from neo4j import GraphDatabase

# Replace these values with your actual Neo4j cloud connection details
uri = "neo4j+s://396fd47c.databases.neo4j.io:7687"
user = "neo4j"
password = "your_password"

# Create the driver and establish a session
driver = GraphDatabase.driver(uri, auth=(user, password))

def test_connection():
    with driver.session() as session:
        greeting = session.run("RETURN 'Connection successful!' AS message")
        for record in greeting:
            print(record["message"])

# Test the connection
test_connection()

# Close the driver when done
driver.close()


def get_topics():
    with driver.session() as session:
        result = session.run("MATCH (t:Topic) RETURN t.name AS topic")
        topics = [record["topic"] for record in result]
        return topics

topics = get_topics()
print("Topics:", topics)


def get_subtopics(topic_name):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (t:Topic {name: $topic_name})-[:HAS_SUBTOPIC]->(st:Subtopic)
            RETURN st.name AS subtopic
            """,
            topic_name=topic_name
        )
        subtopics = [record["subtopic"] for record in result]
        return subtopics

subtopics = get_subtopics("YourTopicName")
print("Subtopics for YourTopicName:", subtopics)



def get_subsubtopics(subtopic_name):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (st:Subtopic {name: $subtopic_name})-[:HAS_SUBSUBTOPIC]->(sst:Subsubtopic)
            RETURN sst.name AS subsubtopic
            """,
            subtopic_name=subtopic_name
        )
        subsubtopics = [record["subsubtopic"] for record in result]
        return subsubtopics

subsubtopics = get_subsubtopics("YourSubtopicName")
print("Sub-subtopics for YourSubtopicName:", subsubtopics)


def get_categories(subsubtopic_name):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (sst:Subsubtopic {name: $subsubtopic_name})-[:HAS_CATEGORY]->(c:Category)
            RETURN c.name AS category
            """,
            subsubtopic_name=subsubtopic_name
        )
        categories = [record["category"] for record in result]
        return categories

categories = get_categories("YourSubsubtopicName")
print("Categories for YourSubsubtopicName:", categories)


def get_nodes(category_name):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Category {name: $category_name})-[:HAS_NODE]->(n:Node)
            RETURN n.name AS node
            """,
            category_name=category_name
        )
        nodes = [record["node"] for record in result]
        return nodes

nodes = get_nodes("YourCategoryName")
print("Nodes for YourCategoryName:", nodes)
