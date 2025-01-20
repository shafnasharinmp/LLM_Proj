import pandas as pd
import google.generativeai as genai
import os
import os
from dotenv import load_dotenv , find_dotenv


load_dotenv()

secret = os.getenv("GEMINI_API_KEY")
print(secret)

genai.configure(api_key=secret)

def dataframe_to_string(file_path):
    """Convert DataFrame to a string format suitable for model input."""
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension == ".xlsx" or file_extension == ".xls":
        df = pd.read_excel(file_path)
    elif file_extension == ".json":
        df = pd.read_json(file_path)
    elif file_extension == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError("Unsupported file format")

    df_string = df.to_string(index=False) 
    return df_string

def generate_gemini_content(df_string):
    prompt = "Analyze the following data and provide insights.:"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"{prompt}\n\n{df_string}")
    return response.text


df_string = dataframe_to_string("input.csv")
insights = generate_gemini_content(df_string)
print(insights)