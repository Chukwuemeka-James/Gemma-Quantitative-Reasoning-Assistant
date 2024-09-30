import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

# Set page config with a modern and professional layout
st.set_page_config(
    page_title="Gemma Quantitative Reasoning Assistant",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("📐 Gemma Quantitative Reasoning Assistant")
st.subheader("Solve Complex Math and Reasoning Problems with AI")
st.markdown(
    """
    Welcome to the **Gemma Quantitative Reasoning Assistant**! This tool helps you solve mathematical 
    and reasoning problems effortlessly. Just enter your mathematical expression or text, and let Gemma do the work.
    """)

# Sidebar for API Key and Additional Information
with st.sidebar:
    st.header("Configuration")
    st.markdown("🔑 **API Configuration**")
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        help="Enter your Groq API key to access the Gemma model."
    )

    st.markdown("💡 **Usage Guide**")
    st.markdown(
        """
        - Input: Provide a math equation or quantitative reasoning text.
        - Summary: The result will be presented below.
        """
    )

# Check if the API key is provided
if groq_api_key:
    # Input Section for Problem Solving
    st.markdown("### Input Problem for Quantitative Reasoning")
    problem_input = st.text_area(
        "Enter your math problem or reasoning question:",
        placeholder="e.g. Solve for x: 2x + 3 = 7",
        height=150
    )

    # Gemma Model Setup
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

    # Prompt template for problem-solving
    prompt_template = """
    Solve the following problem step by step:
    Problem: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Summarization Chain
    if st.button("Solve Problem"):
        if not problem_input.strip():
            st.error("Please enter a math problem to continue.")
        else:
            try:
                with st.spinner("Solving the problem, please wait..."):
                    # Use LLM to solve the problem
                    chain = LLMChain(llm=llm, prompt=prompt)
                    result = chain.run(problem_input)
                    # Display the result
                    st.success("Solution:")
                    st.write(result)

            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.info("Please enter your Groq API key to get started.")

# Footer Section
st.markdown("---")
st.markdown(
    """
    Powered by [Gemma2](https://groq.com) and [LangChain](https://langchain.com). 
    Built with [Streamlit](https://streamlit.io).
    """
)
