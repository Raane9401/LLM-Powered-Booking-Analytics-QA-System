import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sqlite3
import time
from functools import wraps
import pandas as pd
from datetime import datetime
import json
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llm_utils import get_openai_response, get_remote_llm_response
import os
from langchain.schema import Document
from qdrant_client.http.models import PointStruct
from openai import OpenAI
import requests
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

def update_summary_statistics_in_vectordb(client, collection_name, embeddings):
    """Update summary statistics in the vector database."""
    try:
        # Load and process the CSV data
        df = pd.read_csv('cleaned_hotel_bookings.csv')
        summary_stats = {
            'total_bookings': len(df),
            'average_revenue': float(df['revenue'].mean()),
            'cancellation_rate': float(df['is_canceled'].mean() * 100),
            'top_countries': df['country'].value_counts().head(5).to_dict(),
            'average_lead_time': float(df['lead_time'].mean())
        }

        # Create text representation of summary stats
        summary_text = (
            f"Hotel Bookings Summary Statistics:\n"
            f"Total number of bookings: {summary_stats['total_bookings']}\n"
            f"Average revenue per booking: ${summary_stats['average_revenue']:.2f}\n"
            f"Cancellation rate: {summary_stats['cancellation_rate']:.1f}%\n"
            f"Average lead time: {summary_stats['average_lead_time']:.1f} days\n"
            f"Top booking countries: {', '.join(f'{k}: {v}' for k, v in summary_stats['top_countries'].items())}"
        )

        # Get embedding for the summary
        summary_embedding = embeddings.embed_documents([summary_text])[0]
        
        # Store in Qdrant with a special point ID
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=1,  # Fixed ID for summary stats
                    vector=summary_embedding,
                    payload={
                        "content": summary_text,
                        "type": "summary_statistics",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": summary_stats
                    }
                )
            ]
        )
        print("Successfully updated summary statistics in vector database")
        return True
    except Exception as e:
        print(f"Error updating summary statistics: {str(e)}")
        return False

# Configure Flask app with proper static file handling
app = Flask(__name__, static_folder='visualizations', static_url_path='/visualizations')
CORS(app)  # Enable CORS for all routes

# Ensure visualization directory exists
os.makedirs('visualizations', exist_ok=True)

# Initialize embeddings model and get dimension
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Get embedding dimension by embedding a test string
test_embedding = embeddings.embed_query("test")
embedding_dimension = len(test_embedding)

# Initialize Qdrant client and vector store
try:
    # Initialize Qdrant client with REST API
    client = QdrantClient(
        url="http://localhost:6333",
        prefer_grpc=False  # Ensure REST API is used
    )
    collection_name = "hotel_bookings_db"
    
    print("\n[1/3] Checking Qdrant connection...")
    try:
        # Simple health check by getting collections list
        collections = client.get_collections()
        print("✓ Qdrant connection successful")
    except Exception as e:
        print("✗ Error connecting to Qdrant. Make sure it's running on localhost:6333")
        print(f"Error details: {str(e)}")
        raise

    print("\n[2/3] Setting up collection...")
    try:
        # Check if collection exists, if not create it
        try:
            collection_info = client.get_collection(collection_name)
            print(f"✓ Using existing collection '{collection_name}'")
            print(f"Current points in collection: {collection_info.points_count}")
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
            )
            print(f"✓ Created new collection '{collection_name}'")
            
        # Initialize QdrantVectorStore
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        print("✓ Vector store initialized")
        
    except Exception as e:
        print(f"✗ Error with collection: {str(e)}")
        raise

    print("\n[3/3] Processing data...")
    try:
        # Load CSV and check if we need to process data
        df = pd.read_csv('cleaned_hotel_bookings.csv')
        collection_info = client.get_collection(collection_name)
        
        if collection_info.points_count == 0:
            print("Processing and storing data...")
            # Process in smaller batches
            batch_size = 100  # Smaller batch size for better reliability
            total_processed = 0
            
            for i in range(0, len(df), batch_size):
                batch = df[i:i + batch_size]
                
                # Create text for this batch
                texts = []
                for _, row in batch.iterrows():
                    text = (
                        f"Hotel: {row['hotel']}\n"
                        f"Status: {'Canceled' if row['is_canceled'] else 'Confirmed'}\n"
                        f"Lead Time: {row['lead_time']} days\n"
                        f"Revenue: ${row.get('revenue', 0):.2f}\n"
                        f"Country: {row.get('country', '')}"
                    )
                    texts.append(text)
                
                try:
                    # Get embeddings
                    embeddings_batch = embeddings.embed_documents(texts)
                    
                    # Create points
                    points = [
                        PointStruct(
                            id=total_processed + idx,
                            vector=embedding,
                            payload={
                                "content": text,
                                "metadata": {
                                    "timestamp": datetime.now().isoformat()
                                }
                            }
                        )
                        for idx, (text, embedding) in enumerate(zip(texts, embeddings_batch))
                    ]
                    
                    # Upload batch
                    client.upsert(collection_name=collection_name, points=points)
                    
                    total_processed += len(points)
                    progress = (total_processed / len(df)) * 100
                    print(f"\rProgress: {progress:.1f}% ({total_processed}/{len(df)} records)", end="")
                    
                except Exception as batch_error:
                    print(f"\nError processing batch: {str(batch_error)}")
                    continue
            
            print("\n✓ Data processing complete")
            
            # Update summary statistics
            update_success = update_summary_statistics_in_vectordb(client, collection_name, embeddings)
            if update_success:
                print("✓ Summary statistics updated")
            else:
                print("✗ Failed to update summary statistics")
        else:
            print(f"✓ Collection already contains {collection_info.points_count} points")
            
    except FileNotFoundError:
        print("✗ Error: cleaned_hotel_bookings.csv not found")
        raise
    except Exception as e:
        print(f"✗ Error processing data: {str(e)}")
        raise

    print("\nSetup complete! System is ready.")

except Exception as e:
    print(f"\n✗ Fatal error: {str(e)}")
    print("\nPlease ensure:")
    print("1. Qdrant is running on localhost:6333")
    print("2. cleaned_hotel_bookings.csv exists")
    print("3. All required packages are installed")
    raise

# Performance monitoring decorator
def measure_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        
        # Log performance metrics
        with open('api_performance.log', 'a') as log:
            log.write(f"{datetime.now()}, {f.__name__}, {execution_time:.4f}s\n")
        
        if isinstance(result, tuple):
            response, code = result
            if isinstance(response, dict):
                response['execution_time'] = f"{execution_time:.4f}s"
            return response, code
        return result
    return wrapper

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask')
def ask_page():
    return render_template('ask.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/static/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory(app.config['VISUALIZATIONS_FOLDER'], filename)

@app.route('/api/analytics', methods=['GET', 'POST'])
@measure_time
def get_analytics():
    try:
        # Try to connect to SQLite database
        try:
            conn = sqlite3.connect('hotel_analytics.db')
            cursor = conn.cursor()
            
            # Get all visualizations metadata
            cursor.execute("SELECT title, description, file_path, created_at FROM visualizations")
            visualizations = cursor.fetchall()
            conn.close()
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            visualizations = []
        
        # Try to get summary statistics and update vector database
        try:
            df = pd.read_csv('cleaned_hotel_bookings.csv')
            summary_stats = {
                'total_bookings': len(df),
                'average_revenue': float(df['revenue'].mean()),
                'cancellation_rate': float(df['is_canceled'].mean() * 100),
                'top_countries': df['country'].value_counts().head(5).to_dict(),
                'average_lead_time': float(df['lead_time'].mean())
            }

            # Update vector database with new statistics
            update_success = update_summary_statistics_in_vectordb(client, collection_name, embeddings)
            if not update_success:
                print("Warning: Failed to update summary statistics in vector database")

        except Exception as csv_error:
            print(f"CSV error: {str(csv_error)}")
            summary_stats = {
                'error': 'Could not load summary statistics'
            }
        
        # Format the results and clean up file paths
        analytics_report = {
            'summary_stats': summary_stats,
            'visualizations': [
                {
                    'title': row[0],
                    'description': row[1],
                    'file_path': os.path.basename(row[2]),  # Only use the filename
                    'created_at': row[3]
                }
                for row in visualizations
            ] if visualizations else []
        }
        
        return jsonify(analytics_report), 200
        
    except Exception as e:
        print(f"Error in get_analytics: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_llm_response(query, context):
    """Get response from Ollama using the provided context"""
    try:
        # Create a more structured prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""
            Based on the following context, please answer the question.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer: Let me provide a clear response based on the given context."""
        )
        
        # Create LLMChain
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )
        
        # Get response
        response = chain.run({
            "context": context,
            "query": query
        })
        
        if not response:
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        return response.strip()
        
    except Exception as e:
        print(f"Error in get_llm_response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

# Update the answer_question route with better error handling
@app.route('/api/ask', methods=['GET', 'POST'])
@measure_time
def answer_question():
    if request.method == 'GET':
        return jsonify({
            'message': 'Please send a POST request with a query',
            'example': {'query': 'What is the average revenue for hotel bookings?'}
        })
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        print(f"\nReceived query: {query}")
        
        # Verify Qdrant connection
        try:
            collection_info = client.get_collection(collection_name)
            print(f"Connected to collection with {collection_info.points_count} points")
        except Exception as e:
            print(f"Error connecting to Qdrant: {str(e)}")
            return jsonify({
                'error': 'Vector store connection error',
                'answer': "Sorry, I'm having trouble accessing the database."
            }), 500

        # Get context
        context = get_relevant_context(query, qdrant)
        if not context:
            print("No relevant context found")
            return jsonify({
                'answer': "I couldn't find specific information to answer your question about hotel bookings."
            }), 200

        print("\nContext retrieved:", context[:200] + "..." if len(context) > 200 else context)

        # Get LLM response using the new format
        try:
            # Create prompt template
            prompt = PromptTemplate(
                input_variables=["context", "query"],
                template="""
                Based on the following context about hotel bookings, please answer the question.
                
                Context:
                {context}
                
                Question: {query}
                
                Please provide a specific answer using the information from the context:"""
            )
            
            # Use the new chain format
            chain = prompt | llm
            
            # Get response
            response = chain.invoke({
                "context": context,
                "query": query
            })
            
            print("\nLLM Response:", response[:200] + "..." if len(response) > 200 else response)
            
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return jsonify({
                'error': 'LLM error',
                'answer': "Sorry, I encountered an error while generating the response."
            }), 500

        result = {
            'answer': response,
            'execution_time': None  # Will be added by decorator
        }
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        return jsonify({
            'error': str(e),
            'answer': "Sorry, there was an error processing your request."
        }), 500

# Add this new route to your existing Flask app
@app.route('/health-check')
def health_check():
    try:
        # Check Qdrant connection by getting collection info
        try:
            collection_info = client.get_collection(collection_name)
            qdrant_status = "healthy"
        except Exception:
            qdrant_status = "unhealthy"
        
        # Check Ollama connection
        try:
            response = llm.invoke("test")
            ollama_status = "healthy"
        except Exception:
            ollama_status = "unhealthy"
        
        health_status = {
            'status': 'healthy' if qdrant_status == "healthy" and ollama_status == "healthy" else "unhealthy",
            'timestamp': datetime.now().isoformat(),
            'services': {
                'qdrant': qdrant_status,
                'llm': ollama_status
            }
        }
        return jsonify(health_status), 200
    except Exception as e:
        error_status = {
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'services': {
                'qdrant': 'unknown',
                'llm': 'unknown'
            }
        }
        return jsonify(error_status), 500

# Error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# First, add this function before the route definitions
def get_relevant_context(query, qdrant, max_tokens=3000):
    """Get relevant context from vector store for the given query"""
    try:
        print("Fetching context from vector store...")
        
        # First try to get summary statistics
        try:
            summary_results = qdrant.similarity_search_with_score(
                query,
                k=1,
                filter={
                    "must": [
                        {"key": "metadata.type", "match": {"value": "summary_statistics"}}
                    ]
                }
            )
            print(f"Found {len(summary_results)} summary results")
        except Exception as e:
            print(f"Error fetching summary statistics: {str(e)}")
            summary_results = []

        # Get general context
        try:
            general_results = qdrant.similarity_search_with_score(
                query,
                k=5
            )
            print(f"Found {len(general_results)} general results")
        except Exception as e:
            print(f"Error fetching general results: {str(e)}")
            general_results = []

        # Combine and sort results
        all_results = []
        if summary_results:
            all_results.extend(summary_results)
        if general_results:
            all_results.extend(general_results)

        if not all_results:
            print("No results found in vector store")
            return ""

        # Sort by relevance score
        sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        
        # Extract content
        context_parts = []
        current_tokens = 0
        
        for doc, score in sorted_results:
            content = doc.page_content
            # Rough token estimation
            tokens = len(content) // 4
            if current_tokens + tokens > max_tokens:
                break
            
            context_parts.append(f"Relevance Score: {score:.4f}\n{content}")
            current_tokens += tokens

        context = "\n\n".join(context_parts)
        print(f"Retrieved context ({current_tokens} estimated tokens)")
        return context

    except Exception as e:
        print(f"Error in get_relevant_context: {str(e)}")
        return ""

def process_and_store_data(df, chunk_size=10000, chunk_overlap=200, batch_size=1000):
    """Process data into chunks and store in vector database efficiently"""
    try:
        # Use smaller chunk and batch sizes to prevent memory issues
        print(f"Processing with chunk_size={chunk_size}, batch_size={batch_size}")
        
        # Clear existing collection to ensure clean state
        try:
            print("\n[1/5] Preparing database...")
            client.delete_collection(collection_name)
            time.sleep(1)  # Wait for deletion to complete
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
            )
            print("✓ Database reset complete")
        except Exception as e:
            print(f"✗ Error resetting collection: {str(e)}")
            return False

        print("\n[2/5] Starting data processing...")
        # Group related bookings
        df['month_year'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'].astype(str))
        total_hotels = df['hotel'].nunique()
        total_months = df['month_year'].nunique()
        total_combinations = total_hotels * total_months
        processed_combinations = 0
        
        print(f"Total hotels: {total_hotels}")
        print(f"Total months: {total_months}")
        print(f"Processing {total_combinations} hotel-month combinations...")
        
        grouped_data = []
        
        # Process data in groups with progress tracking
        for (hotel, month_year), group in df.groupby(['hotel', 'month_year']):
            processed_combinations += 1
            progress = (processed_combinations / total_combinations) * 100
            print(f"\rProcessing groups: {processed_combinations}/{total_combinations} ({progress:.1f}%)", end="")
            
            group_text = f"Hotel: {hotel}, Period: {month_year.strftime('%B %Y')}\n\n"
            
            # Calculate group statistics
            group_stats = {
                'total_bookings': len(group),
                'avg_revenue': group['revenue'].mean(),
                'cancellation_rate': (group['is_canceled'].mean() * 100),
                'avg_lead_time': group['lead_time'].mean(),
                'top_countries': group['country'].value_counts().head(3).to_dict()
            }
            
            group_text += f"Period Summary:\n"
            group_text += f"- Total Bookings: {group_stats['total_bookings']}\n"
            group_text += f"- Average Revenue: ${group_stats['avg_revenue']:.2f}\n"
            group_text += f"- Cancellation Rate: {group_stats['cancellation_rate']:.1f}%\n"
            group_text += f"- Average Lead Time: {group_stats['avg_lead_time']:.1f} days\n"
            group_text += f"- Top Countries: {', '.join(group_stats['top_countries'].keys())}\n\n"
            
            grouped_data.append(group_text)
            
        print("\n✓ Group processing complete")
        
        print("\n[3/5] Chunking data...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        
        all_chunks = []
        total_groups = len(grouped_data)
        for idx, text in enumerate(grouped_data, 1):
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            progress = (idx / total_groups) * 100
            print(f"\rChunking progress: {idx}/{total_groups} ({progress:.1f}%)", end="")
        
        print(f"\n✓ Created {len(all_chunks)} chunks")
        
        print("\n[4/5] Storing chunks in vector database...")
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        total_stored = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_chunks))
            batch = all_chunks[start_idx:end_idx]
            
            try:
                # Create documents with metadata
                documents = []
                for chunk_idx, chunk in enumerate(batch):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "chunk_id": total_stored + chunk_idx,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
                
                # Get embeddings and store in database
                texts = [doc.page_content for doc in documents]
                embeddings_batch = embeddings.embed_documents(texts)
                
                points = [
                    PointStruct(
                        id=doc.metadata["chunk_id"],
                        vector=embedding,
                        payload={
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                    )
                    for doc, embedding in zip(documents, embeddings_batch)
                ]
                
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                total_stored += len(points)
                progress = (batch_idx + 1) / total_batches * 100
                print(f"\rStoring progress: {batch_idx + 1}/{total_batches} batches ({progress:.1f}%) - Total points: {total_stored}", end="")
                
            except Exception as batch_error:
                print(f"\n✗ Error processing batch {batch_idx + 1}: {str(batch_error)}")
                continue
        
        print("\n✓ Storage complete")
        
        # Verify storage and update summary statistics
        print("\n[5/5] Verifying data and updating summary statistics...")
        collection_info = client.get_collection(collection_name)
        print(f"Final points count in database: {collection_info.points_count}")
        
        if collection_info.points_count > 0:
            print("✓ Data processing and storage successful")
            # Update summary statistics
            update_success = update_summary_statistics_in_vectordb(client, collection_name, embeddings)
            if update_success:
                print("✓ Summary statistics updated successfully")
            else:
                print("✗ Failed to update summary statistics")
            return True
        else:
            print("✗ Error: No points stored in vector database")
            return False
            
    except Exception as e:
        print(f"\n✗ Error in process_and_store_data: {str(e)}")
        return False

# Also ensure Ollama is properly initialized at startup
try:
    # Initialize Ollama with more robust error handling
    print("\nInitializing Ollama LLM...")
    try:
        llm = Ollama(
            model="llama2",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_k=10,
            top_p=0.95,
            repeat_penalty=1.1,
            timeout=30  # Add timeout
        )
        # Test the LLM
        test_response = llm.invoke("test")
        if test_response:
            print("✓ Ollama LLM initialized successfully")
        else:
            print("✗ Ollama LLM initialization failed")
    except Exception as e:
        print(f"✗ Error initializing Ollama: {str(e)}")
        print("Please ensure Ollama is running and the llama2 model is installed")
        raise

except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
