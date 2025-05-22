import os
import json
import pandas as pd
import sqlite3
import requests
import dotenv
from pathlib import Path
import streamlit as st
# Load environment variables from .env file if present
try:
    dotenv.load_dotenv()
    print("Environment variables loaded from .env file")
except:
    print("No .env file found or error loading it")

# OpenRouter API Key from environment variable
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
print(f"OPENROUTER_API_KEY exists: {OPENROUTER_API_KEY is not None}")

# Path to the database file (will be created in memory)
DB_NAME = ":memory:"  # Use in-memory database for temporary storage

def create_db_from_xlsx(xlsx_file_path):
    """
    Create a SQLite database from an XLSX file.
    
    Args:
        xlsx_file_path (str): Path to the XLSX file
        
    Returns:
        sqlite3.Connection: Connection to the SQLite database
    """
    if not os.path.exists(xlsx_file_path):
        raise FileNotFoundError(f"XLSX file not found: {xlsx_file_path}")
    
    print(f"Loading XLSX file: {xlsx_file_path}")
    
    # Read XLSX file
    df = pd.read_excel(xlsx_file_path)
    
    # Replace NaN values with None for SQL compatibility
    df = df.replace({pd.NA: None})
    df = df.where(pd.notnull(df), None)
    
    # Create in-memory SQLite database
    conn = sqlite3.connect(DB_NAME)
    
    # Convert column names to SQL-friendly format (replace spaces with underscores)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    
    # Write to SQLite
    df.to_sql("Library", conn, if_exists="replace", index=False)
    
    # Print info about the database
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Library)")
    columns = cursor.fetchall()
    
    print(f"Created database table 'Library' with columns:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    print(f"Loaded {len(df)} records into the database")
    
    return conn

def generate_sql_query(user_query, table_schema):
    """
    Use OpenRouter LLM to generate an SQL query from natural language.
    
    Args:
        user_query (str): The user's natural language query
        table_schema (str): Schema information about the table
        
    Returns:
        str: Generated SQL query or None if unsuccessful
    """
    if not OPENROUTER_API_KEY:
        print("Error: OpenRouter API key not provided in environment variables")
        return None
    
    # System prompt explaining the task and database schema
    system_prompt = f"""
    You are a SQL query generator for a library database. 
    
    The database has a single table called "Library" with the following schema:
    {table_schema}

    Your task is to convert natural language queries about the library into SQL queries.
    
    ALWAYS return ONLY the raw SQL query without any explanation or markdown formatting.
    Return ONLY valid SQLite syntax.
    
    Remeber that Booked columns have value Yes or NULL and Return Date have value Date or NULL
    AUTO CORRECT THE NAMES OF THE BOOKS IF YOU KNOW THEY ARE WRONG!!
    AUTO CORRECT THE SPELLINGS IF YOU KNOW THEY ARE WRONG!!

    Examples:
    
    User: "What books are available?"
    Response: SELECT * FROM Library WHERE Booked IS NULL;
    
    User: "Show me books by J.K. Rowling"
    Response: SELECT * FROM Library WHERE Author = 'J.K. Rowling';
    
    User: "Tell me about Harry Potter"
    Response: SELECT * FROM Library WHERE Book_Name LIKE '%Harry Potter%';
    
    User: "Which books cost less than $15?"
    Response: SELECT * FROM Library WHERE Selling_Price < 15 ORDER BY Selling_Price;
    """
    
    # User prompt with their question
    prompt = f"Convert this library question to SQL: '{user_query}'"
    
    print(f"Sending query to LLM: '{user_query}'")
    
    # Make the API request to OpenRouter
    try:
        print("Making request to OpenRouter API...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        
        print(f"OpenRouter API status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error from OpenRouter API: {response.text}")
            return None
        
        result = response.json()
        sql_query = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Clean up the SQL query (remove markdown formatting if present)
        sql_query = sql_query.strip()
        if sql_query.startswith('```') and sql_query.endswith('```'):
            sql_query = sql_query[3:-3].strip()
        if sql_query.startswith('sql'):
            sql_query = sql_query[3:].strip()
            
        print(f"Generated SQL query: {sql_query}")
        return sql_query
            
    except Exception as e:
        print(f"Error calling OpenRouter API: {str(e)}")
        return None

def execute_sql_query(conn, sql_query):
    """
    Execute an SQL query on the SQLite database.
    
    Args:
        conn (sqlite3.Connection): Connection to the SQLite database
        sql_query (str): The SQL query to execute
        
    Returns:
        list: Query results as a list of dictionaries
    """
    if not sql_query:
        return {"error": "No SQL query provided"}
    
    try:
        print(f"Executing SQL query: {sql_query}")
        
        # Execute query
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Fetch all rows and convert to list of dictionaries
        rows = cursor.fetchall()
        results = []
        for row in rows:
            result = {}
            for i, col in enumerate(column_names):
                result[col] = row[i]
            results.append(result)
        
        print(f"Query returned {len(results)} results")
        return results
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error executing query: {error_msg}")
        return {"error": error_msg}

def get_table_schema(conn):
    """
    Get the schema of the Library table as a formatted string.
    
    Args:
        conn (sqlite3.Connection): Connection to the SQLite database
        
    Returns:
        str: Formatted schema information
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Library)")
    columns = cursor.fetchall()
    
    schema_info = []
    for col in columns:
        name = col[1]
        data_type = col[2]
        schema_info.append(f"- {name} ({data_type})")
    
    return "\n".join(schema_info)

def library_query(conn, user_query):
    """
    Process a natural language query about the library.
    
    Args:
        conn (sqlite3.Connection): Connection to the SQLite database
        user_query (str): Natural language query
        
    Returns:
        dict: Query results and metadata
    """
    print(f"\n=== Processing query: '{user_query}' ===")
    
    # Get the table schema for the LLM prompt
    table_schema = get_table_schema(conn)
    
    # Generate SQL query
    sql_query = generate_sql_query(user_query, table_schema)
    if not sql_query:
        return {"error": "Failed to generate SQL query"}
    
    # Execute SQL query
    results = execute_sql_query(conn, sql_query)
    
    # Return results with metadata
    return {
        "query": user_query,
        "sql": sql_query,
        "results": results
    }

def db_endpoint(user_query):

    xlsx_file_path = "data/raw/books_catalog.xlsx"
    
    # Expand user directory if needed
    xlsx_file_path = os.path.expanduser(xlsx_file_path)
    
    try:
        conn = create_db_from_xlsx(xlsx_file_path)
        print("\nDatabase created successfully. Ready for queries.")
        results = library_query(conn, user_query)
        
        print("\n--- Results ---")
        print(f"Query: {results.get('query')}")
        print(f"SQL: {results.get('sql')}")
        
        if "error" in results:
            print(f"Error: {results['error']}")
        elif "error" in results.get("results", {}):
            print(f"Error: {results['results']['error']}")
        else:
            print("\nResults:")
            data = results.get("results", [])
            if not data:
                print("No results found")
            else:
                for i, item in enumerate(data):
                    print(f"\nItem {i+1}:")
                    for key, value in item.items():
                        print(f"  {key}: {value}")
        
        print("\n" + "-"*50)
        
        # Return the results dictionary
        return results
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e), "query": user_query}
    finally:
        # Close database connection if it exists
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")

# if __name__ == "__main__":
#     user_quer="which books are available"
#     db_endpoint(user_query)