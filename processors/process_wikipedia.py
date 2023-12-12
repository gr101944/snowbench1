import json 
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

def process_wikipedia(prompt, llm):
    print("Processing Wikipedia...", prompt)
    wiki_not_found = "No good Wikipedia Search Result was found in wikipedia"  
    tools = load_tools (["wikipedia"], llm=llm)
    agent = initialize_agent (tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    wiki_research = agent.run(prompt) 

    if (wiki_research == wiki_not_found):
        return_code = 400        
    else:
        return_code = 200
    
        # Create a dictionary to hold the data
    data = {
            "source": "Wikipedia",
            "response": wiki_research,
            "return_code": return_code
    }
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)

    return json_data