from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

# For financial sentiment analysis
finbert = pipeline("sentiment-analysis", 
                  model="ProsusAI/finbert", 
                  tokenizer="ProsusAI/finbert")

# For financial Q&A (keep your existing TAPAS for structured data)
tabqa = pipeline("table-question-answering", 
                model="google/tapas-base-finetuned-wtq")

try:
    # load csv file into pandas dataframe
    data = pd.read_csv("most_active_raw_20250826_120053.csv")
    
    # Example of financial text analysis
    financial_text = "The company reported strong earnings growth and increased dividend."
    sentiment_result = finbert(financial_text)
    print(f"Financial Sentiment: {sentiment_result}")
    
    # Your existing table analysis
    table = data.astype(str)
    question = "What is the volume for TSLA?"
    table_result = tabqa(table=table, query=question)
    print(f"Table Analysis: {table_result}")

except FileNotFoundError:
    print("CSV file not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {str(e)}")