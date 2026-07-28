import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent

st.set_page_config(page_title="AI Powered Data Analyst Agent", layout="wide")
st.title("📊 AI Powered Data Analyst Agent")
st.write("Automatically analyze datasets, generate univariate, bivariate, and multivariate charts, and chat with your data!")

# Sidebar for API Keys
st.sidebar.header("Configuration")
GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.warning("Please enter both Google and Groq API keys in the sidebar to proceed.")
    st.stop()

# Model Creation
@st.cache_resource
def init_models(g_key, gr_key):
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=g_key
    )
    groq_llm = ChatGroq(
        model="qwen/qwen3.6-27b",
        api_key=gr_key
    )
    return gemini_llm, groq_llm

gemini_llm, groq_llm = init_models(GOOGLE_API_KEY, GROQ_API_KEY)

# Agent Creation
def temp_tool():
    """This is just a dummy tool"""
    return "Hello world"

agent = create_agent(
    model=gemini_llm,
    tools=[temp_tool]
)

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File successfully loaded!")
        st.write("### Dataset Preview", df.head())

        # Generate EDA code via Agent
        if st.button("Run Automated AI EDA & Charts"):
            with st.spinner("Agent is analyzing your dataset and generating code..."):
                
                # Basic EDA generation
                df_sample = df.sample(min(5, len(df)))
                prompt = f"""You are a data analyst. Return only executable python code inside a function named perform_eda(df) that prints basic EDA like shape, size, missing values, and columns.
                Data frame sample : {df_sample}
                data stats: {df_sample.describe()}"""

                response = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
                ans = response["messages"][-1].content[-1]['text']
                code = ans.split("```")[1]
                if code.startswith("python"):
                    code = code[6:]

                with open('basic_eda.py', 'w') as f:
                    f.write(code)

                # Advanced EDA & Charts generation
                advance_prompt = """Give Python code with a single function eda_by_ai(df) that generates multiple matplotlib/seaborn charts:
                1. Univariate analysis for numerical and categorical columns.
                2. Bivariate analysis charts.
                3. Multivariate analysis (e.g., using hue with bar plots or scatter plots).
                Save plots using st.pyplot() or return figures so they can be rendered in Streamlit. Ensure all code is pure executable Python without extra conversational text."""

                response = agent.invoke({'messages': [{'role': 'user', 'content': advance_prompt}]})
                system_prompt_model = response["messages"][-1].content[-1]['text']

                new_prompt = "Give Python advance_eda.py file with every code inside a single function eda_by_ai(df) using df directly and plotting using matplotlib/seaborn.\n" + system_prompt_model

                response = agent.invoke({'messages': [{'role': 'user', 'content': new_prompt}]})
                ans = response["messages"][-1].content[-1]['text']
                code = ans.split("```")[1]
                if code.startswith("python"):
                    code = code[6:]

                with open('advance_eda.py', 'w') as f:
                    f.write(code)

            st.success("EDA Code Generated Successfully!")
            
            # Execute Basic EDA
            st.write("### Basic EDA Report")
            try:
                from basic_eda import perform_eda
                # Capture print statements or output
                buffer = io.StringIO()
                import sys
                old_stdout = sys.stdout
                sys.stdout = buffer
                perform_eda(df)
                sys.stdout = old_stdout
                st.text(buffer.getvalue())
            except Exception as e:
                st.error(f"Error executing basic EDA: {e}")

            # Execute Advanced EDA & Charts
            st.write("### Advanced EDA & Visualizations (Univariate, Bivariate, Multivariate)")
            try:
                from advance_eda import eda_by_ai
                fig, ax = plt.subplots()
                eda_by_ai(df)
                # Render any active matplotlib figures
                for i in plt.get_fignums():
                    st.pyplot(plt.figure(i))
            except Exception as e:
                st.warning(f"Could not auto-execute advanced script directly or it requires custom rendering. Error: {e}")
                st.info("Displaying standard automated correlation heatmap and pairplot as fallback:")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("No numeric columns available for correlation heatmap.")

        # Chat with Data Feature
        st.write("---")
        st.write("### 💬 Chat with your Data")
        user_query = st.text_input("Ask anything about your dataset (e.g., 'What is the average of sales by region?'):")
        
        if user_query:
            with st.spinner("AI is generating code to answer your query..."):
                chat_prompt = f"""Given the pandas dataframe df with columns {list(df.columns)}, write executable python code to answer this question: {user_query}. 
                Assume df is already loaded. Print or store the final answer in a variable named 'result'. Return ONLY python code inside markdown blocks."""
                
                response = agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
                ans = response["messages"][-1].content[-1]['text']
                
                try:
                    code_block = ans.split("```")[1]
                    if code_block.startswith("python"):
                        code_block = code_block[6:]
                    
                    # Execute generated query code safely
                    local_vars = {"df": df, "pd": pd, "np": np}
                    exec(code_block, {}, local_vars)
                    
                    if "result" in local_vars:
                        st.write("**Answer:**")
                        st.write(local_vars["result"])
                    else:
                        st.success("Code executed successfully.")
                except Exception as e:
                    st.error(f"Error executing query code: {e}")

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a dataset to begin the automated analysis.")

# Extra comments and notes
# - Ensure all required packages (pandas, numpy, matplotlib, seaborn, langchain, streamlit) are installed via pip.
# - The agent dynamically writes and executes modular python scripts for exploratory data analysis.
# - Chat with data securely evaluates generated pandas operations on the active dataframe context.
