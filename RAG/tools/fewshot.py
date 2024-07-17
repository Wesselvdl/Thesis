from langchain.chains import GraphCypherQAChain
# tag::import-prompt-template[]
from langchain.prompts.prompt import PromptTemplate
# end::import-prompt-template[]

from solutions.llm import llm
from solutions.graph import graph

# tag::prompt[]
CYPHER_GENERATION_TEMPLATE = """
Agent Role: You are an expert Neo4j Developer who translates user questions into Cypher queries to answer questions about datacenter configurations and provide detailed insights.

Agent Instructions:
1. Interpret the User Query: Determine the specific area or topic within the datacenter based on the users question.
2. Construct the Cypher Query: Formulate the Cypher query to fetch relevant nodes based on the `PartOf` relationship path from the specified `Category` to related `Rule` nodes.
3. Present the Results: List queried information of the retrieved `Rule` nodes clearly, ensuring the response is easy to understand and directly addresses the users query.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
------------------------------------------------------------------------
Schema Details:
- Node Types: 
    + 'Regulation' with Node Property 'Id', 'Description', 'Name', 'Value', 'Notes', 'Referred Documentation',
    + 'Category' with Node Property 'Category'
    + 'Section' with Node Property 'Section'
    + 'SubSection' with Node Property 'SubSection'
- Relationships: 
    + 'Has_Category' ('Regulation' --> 'Category') 
    + 'Has_Section' ('Regulation' --> 'Section') 
    + 'Has_SubSection' ('Regulation' --> 'SubSection') 

- Unique Property Values for Category:'Category', 'Entire Data Centre', 'New build or retrofit', 'Optional', 'New IT Equipment', 'New Software', 'New IT Equipment and New build or retrofit'
- Unique Property Values for Section: 'Data Centre Utilisation, 'Management and Planning', 'IT Equipment and Services', 'Cooling', 'Data Centre Power Equipment', 'Data Centre Building', 'Monitoring', 'Other Data Centre Equipment'
- Unique Property Values for SubSection: 'Involvement of Organisational Groups', 'General Policies', 'Resilience Level and Provisioning', 'Selection and Deployment of New IT Equipment', 'Deployment of New IT Services', 'Management of Existing IT Equipment and Services', 'Data Management', 'Air Flow Management and Design', 'Cooling Management', 'Temperature and Humidity Settings', 'Free Cooling / Economised Cooling', 'High Efficiency Cooling Plant', 'Computer Room Air Conditioners / Air Handlers', 'Direct Liquid Cooling', 'Reuse of Data Centre Waste Heat', 'Selection and Deployment of New Power Equipment', 'Management of Existing Power Equipment', 'General Practices', 'Building Physical Layout', 'Building Geographic Location', 'Water sources', 'Energy Use and Environmental Performance Measurement', 'Energy Use and Environmental Data Collection and Performance Logging', 'Energy Use and Environmental Performance Reporting', 'IT Reporting'
------------------------------------------------------------------------
Query Formatting Rules:
- Use only the provided relationship types and node properties in the schema.
- Do not use any other relationship types or node properties that are not provided.
- EVERY SINGLE TIME THAT THE WORD 'Sub-Section' APPEARS IN A NODE OR RELATIONSHIP, IT SHOULD BE ENCLOSED IN BACKTICKS.
------------------------------------------------------------------------
EXAMPLE 1
User Question: `List all the descriptions of the Regulations with IDs; 3.2.8, 4.2.1, 5.1.1 & 5.4.1.1?`
Cypher Query:
```
MATCH (reg:Regulation)
WHERE reg.ID IN {{['3.2.8', '4.2.1', '5.1.1', '5.4.1.1']}}
RETURN reg.ID AS RegulationID, reg.Description AS RegulationDescription
```
EXAMPLE 2
User Question: `List all the descriptions of the Regulations that are related to the Section 'Cooling'.`
Cypher Query:
```
MATCH (sec:Section {{Section: 'Cooling'}})<-[:Has_Section]-(reg:Regulation)
RETURN reg.ID AS RegulationID, reg.Description AS RegulationDescription
```
EXAMPLE 3
User Question: `List all the IDs of the Regulations that are related to the Category 'Entire Data Centre'.`
Cypher Query:
```
MATCH (cat:Category {{Category: 'Entire Data Centre'}})<-[:Has_Category]-(reg:Regulation)
RETURN reg.ID AS RegulationID
```
EXAMPLE 4
User Question: `Generate an action plan for the regulations with IDs 3.2.8, 4.2.1, and 5.1.1.`
Cypher Query:
```
MATCH (reg:Regulation)
WHERE reg.ID IN {{['3.2.8', '4.2.1', '5.1.1']}}
RETURN reg.ID AS RegulationID, reg.Description AS RegulationDescription, reg.Value AS RegulationValue, reg.Notes AS RegulationNotes, reg.`Referred Documentation` AS RegulationDocumentation
```
EXAMPLE 2
User Question: `List all the descriptions of the Regulations that are related to the Section 'Air Flow Management and Design'.`
Cypher Query:
```
MATCH (Sec:Section {{Section: 'Air Flow Management and Design'}})<-[:Has_SubSection]-(reg:Regulation)
RETURN reg.ID AS RegulationID, reg.Description AS RegulationDescription
```
------------------------------------------------------------------------

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

QA_PROMPT = """
User Query Scenario: A user asks about topics related to specific regulatory IDs, categories, sections, or sub-sections of regulations. It is your goal to provide a clear answer based on the provided Context. There are multiple ways users may ask questions:

1. Listing all IDs and a one-sentence description related to a certain Category, Section, or Sub-Section.
2. Generating an action plan from the descriptions without using technical language if explicitly asked.

Follow the guidelines below to provide helpful answers:

1. If the user asks to list IDs and descriptions:
   - Provide the IDs and a concise one-sentence description for each related regulation.

2. If the user explicitly asks for an action plan:
   - Rewrite the descriptions into a clear, step-by-step action plan, avoiding any technical language.
----------------------------------------------------------------------------------------------------------
Examples:
User Query 1: List all the IDs of the Regulations that are related to the Category 'Entire Data Centre'.
Expected Context: [{{'RegulationID': '3.2.8'}}, {{'RegulationID': '4.2.1'}}, {{'RegulationID': '5.1.1'}}]
Helpful Answer: All the regulations related to the Category 'Entire Data Centre' are: 
- 3.2.8
- 4.2.1
- 5.1.1

User Query 2: Generate an action plan for the regulations with IDs 3.2.8, 4.2.1, and 5.1.1.
Expected Context: [{{'RegulationID': '3.2.8', 'RegulationDescription': 'Ensure data encryption at rest and in transit.', 'RegulationValue':'4','RegulationNotes':'Use Data',RegulationDocumentation:'ISO14044'}}, {{'RegulationID': '4.2.1', 'RegulationDescription': 'Implement access controls based on user roles.', 'RegulationValue':'4','RegulationNotes':'',RegulationDocumentation:''}}, {{'RegulationID': '5.1.1', 'RegulationDescription': 'Conduct regular security audits and vulnerability assessments.', 'RegulationValue':'4','RegulationNotes':'',RegulationDocumentation:'ISO14044'}}]
Helpful Answer: 
Action Plan:
- 3.2.8: Ensure all data is encrypted both at rest and during transit to protect against unauthorized access. 4
    + Notes: Use Data
    + Referred Documentation: ISO14044
- 4.2.1: Implement strict access controls based on user roles to ensure only authorized personnel have access to sensitive information. 4
- 5.1.1: Conduct regular security audits and vulnerability assessments to identify and mitigate potential security risks. 4
    + Referred Documentation: ISO14044

User Query 3: List all the descriptions of the Regulations that are related to the Section 'Cooling'.
Expected Context: [{{'RegulationID': '3.2.8', 'RegulationDescription': 'Ensure data encryption at rest and in transit.'}}, {{'RegulationID': '4.2.1', 'RegulationDescription': 'Implement access controls based on user roles.'}}, {{'RegulationID': '5.1.1', 'RegulationDescription': 'Conduct regular security audits and vulnerability assessments.'}}]
Helpful Answer: 
- 3.2.8: Ensure data encryption at rest and in transit.
- 4.2.1: Implement access controls based on user roles.
- 5.1.1: Conduct regular security audits and vulnerability assessments.
----------------------------------------------------------------------------------------------------------

If the provided information is empty, say that you don't know the answer.

Information:
{context}

Question:
{question}

Helpful Answer:
"""
# end::prompt[]

# tag::template[]
cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
# end::template[]

qa_prompt = PromptTemplate.from_template(QA_PROMPT)

# tag::cypher-qa[]
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt
)
# end::cypher-qa[]