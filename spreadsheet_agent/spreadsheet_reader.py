import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from litellm import completion
import os
from datetime import datetime

class QuestionInput(BaseModel):
    question: str = Field(..., description="The question to ask about the CSV data")
    context_window: Optional[int] = Field(default=1000, description="Number of characters to include in context")
    model: Optional[str] = Field(default="gpt-3.5-turbo", description="Model to use for completion"),
    csv_path: Optional[str] = Field(default="weekly-financials.csv", description="Path to the CSV file")

def read_csv_data(file_path: str) -> pd.DataFrame:
    """
    Read and validate CSV data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

def prepare_context(df: pd.DataFrame, file_name: str, context_window: int) -> str:
    """
    Prepare context from DataFrame for the LLM
    """
    # Convert DataFrame info to string
    context = f"File Name: {file_name}\n"
    context += f"CSV Columns: {', '.join(df.columns)}\n"
    context += f"Number of rows: {len(df)}\n"
    context += "Sample data:\n"
    
    # Add sample data (first few rows)
    sample_data = df.head(3).to_string()
    if len(sample_data) > context_window:
        sample_data = sample_data[:context_window] + "..."
    
    context += sample_data
    return context

def answer_question(question_input: QuestionInput) -> str:
    """
    Process a question about CSV data using LiteLLM
    """
    # Read CSV
    df = read_csv_data(question_input.csv_path)

    # Parse the name of the file from the path
    file_name = question_input.csv_path.split("/")[-1]
    
    # Prepare context
    context = prepare_context(df, file_name, question_input.context_window)
    
    # Construct prompt
    prompt = f"""Given the following CSV data:

{context}

Question: {question_input.question}

Please provide a clear and concise answer based on the data provided."""

    # Get completion from LiteLLM
    try:
        response = completion(
            model=question_input.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes CSV data and answers questions about it. List the name of the file used for reference in your response."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting completion: {str(e)}")

def main():
    # Example usage
    
    # Define question using Pydantic model
    question = QuestionInput(
        question="What was the highest revenue week?",
        context_window=1500,
        model="gpt-3.5-turbo",
        csv_path="spreadsheet_agent/weekly-financials.csv"
    )
    
    try:
        answer = answer_question(question)
        print(f"\nQuestion: {question.question}")
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()