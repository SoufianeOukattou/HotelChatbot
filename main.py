import os.path
import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# Fetch the API key from the environment variable
# api_key = os.getenv('OPENAI_API_KEY')

from llama_index.core import Settings
from llama_index.core import PromptTemplate

Settings.chunk_size = 512

# Local settings
from llama_index.core.node_parser import SentenceSplitter


from llama_index.core import (
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)



custom_prompt = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)


PERSIST_DIR = "./storage"

# Side bar contents
with st.sidebar:
    st.title('Achbal Artificial Intelligence')
    st.markdown('''
    ## More Info:
    - [Achbal Mail](mailto:achbal.business@gmail.com)
                
         This application communicates with Artificial Intelligence Ashbal powered by (LLM), built using:
    - [Streamlit](https://streamlit.io/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    - [Idarati dataset](https://www.idarati.ma/)
    ''')
    add_vertical_space(5)
    st.write('Created by Ashbal Team')

st.title('Communicate with Artificial Intelligence Ashbal')

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Al Mamounia Hotel Reservations. I'm here to assist you with any inquiries you may have. How can I help you today?"}
    ]

@st.cache_resource(show_spinner=False)
def load_index():
    with st.spinner(
        text="Loading Al Mamounia Hotel documents... Please wait! This may take 1 to 2 minutes."
    ):
        if not os.path.exists("./storage"):
            documents = SimpleDirectoryReader("data").load_data()
            index = VectorStoreIndex.from_documents(
                    documents, transformations=[SentenceSplitter(chunk_size=512)]

                )
            index.storage_context.persist()
        else:
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
          
            index = load_index_from_storage(storage_context)
        return index

index = load_index()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

prompt = st.chat_input("Say something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Concatenate user's input with previous messages
                conversation_history = " ".join(
                    [msg["content"] for msg in st.session_state.messages]
                )
                response = st.session_state.chat_engine.chat(conversation_history)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)  # Add response to message history
            except Exception as e:
                st.error(
                    "!Free API key requests have been exhausted. Please consider waiting for a minute"
                )
                st.error(
                    "!In the meantime, feel free to contact the developers for any questions or support for this project"
                )
