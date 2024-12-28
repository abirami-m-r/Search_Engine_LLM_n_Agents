import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")


api_wrap = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrap)

api_wrap = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrap)

search = DuckDuckGoSearchRun(name="Search")


st.title("Langchain - CHat with Search using Tools and Agents")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("ENter ur Groq API",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"Assisstant",
            "content":"Hi, I'm a chatbot who can search the web"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({"role":"user",
                                      "content": prompt})
    st.chat_message("user").write(prompt)
    llm = ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192",streaming=True)
    tools = [search,arxiv,wiki]

    ##Zero shot = impulsive, direct to the qn
    ##Chat_wero shot = ans based on chat convo
    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handling_parsing_errors=True)
    
    with st.chat_message("assisstant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,
                                    callbacks =[st_cb])
        st.session_state.messages.append({"role":"assisstant",
                                          "content":response})
        st.write(response)
    
