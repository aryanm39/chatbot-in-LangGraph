# comment out -> subgraph 
# not commented -> subgraph - shared 

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str

# class SubState(TypedDict):
#     input_text: str
#     translated_text: str

parent_llm = ChatOpenAI(model='gpt-4o-mini')
subgraph_llm = ChatOpenAI(model='gpt-4o')  

def translate_text(state: ParentState):    # def translate_text(state: SubState):
    prompt = f"""
        Translate the following text to Hindi.
        Keep it natural and clear. Do not add extra content.

        Text:
        {state["answer_eng"]}  
        """.strip()   # {state["input_text"]}
    
    translated_text = subgraph_llm.invoke(prompt).content
    return {'answer_hin': translated_text}   #     return {'translated_text': translated_text}

subgraph_builder = StateGraph(ParentState)    # subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node('translate_text', translate_text)
subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)
subgraph = subgraph_builder.compile()

def generate_answer(state: ParentState):
    answer = parent_llm.invoke(f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}").content
    return {'answer_eng': answer}

# def translate_answer(state: ParentState):
#     result = subgraph.invoke({'input_text': state['answer_eng']})
#     return {'answer_hin': result['translated_text']}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("answer", generate_answer)

#remove below line for subgraph 
parent_builder.add_node("translate", subgraph)

# comment out below line for subgraph
# parent_builder.add_node("translate", translate_answer)

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)
graph = parent_builder.compile()
graph
graph.invoke({'question': 'What is quantum physics'})