
import streamlit as st
from Graphs.FilterRes import run_workflow_filter




from PIL import Image



st.set_page_config(layout="wide")
# You can always call this function where ever you want

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="logo.png", width=170, height=170)




st.image(my_logo)



def on_chat_submit(chat_input):
    """
    Handle chat input submissions and interact with the OpenAI API.

    Parameters:
        chat_input (str): The chat input from the user.
        api_key (str): The OpenAI API key.
        latest_updates (dict): The latest Streamlit updates fetched from a JSON file or API.
        use_langchain (bool): Whether to use LangChain OpenAI wrapper.

    Returns:
        None: Updates the chat history in Streamlit's session state.
    """
    

    # Initialize the OpenAI API
  

    # Initialize the conversation history with system and assistant messages
    if 'conversation_history' not in st.session_state:
        
        
        # Initialize conversation_history
        st.session_state.conversation_history = [
            {"role": "user", "content": chat_input}
        ]

    try:
        # Logic for assistant's reply
        assistant_reply = ""

        if chat_input:
            assistant_reply = run_workflow_filter(chat_input)

        

        # Update the Streamlit chat history
        if "history" in st.session_state:
            st.session_state.history.append({"role": "user", "content": chat_input})
            st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        # Error handling
        error_message = f"An error occurred: {e}"

        #st.session_state.history.append({"role": "assistant", "content": error_message})




# Set page title
st.title('FEYNMAN GRAPH 1.0')


st.header('Primer sistema de IA especializado en mentorias de fisica para tempranas edades', divider='blue')




        

message = st.chat_input("Escribe tu mensaje aquí...")
if "history" not in st.session_state:
        st.session_state.history = []
if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []





with st.container():
    messages = st.container()
    
    if message:
        

        on_chat_submit(message)
        for message in st.session_state.history[-20:]:
            role = message["role"]
            with st.chat_message(role):
                st.write(message["content"])


    

        
            

    
  
            
        
        
        
        
        
            
        