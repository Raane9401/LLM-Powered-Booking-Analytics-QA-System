# Hotel Booking Analytics System

This application provides comprehensive analytics for hotel booking data, including data preprocessing, visualization, and a query interface powered by qdrant vector db and BAAI Embedings search and LLM integration.


<p align="center">
  <video src="DemoVideo.gif" width="500px"></video>
</p>

  

## Features

- Data cleaning and preprocessing of hotel booking records
- Visualization of key metrics (revenue trends, cancellation rates, geographical distribution)
- Vector database storage for semantic search capabilities
- REST API for accessing analytics and querying the data
- LLM integration for natural language question answering


## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- Llama 2 Installed with Ollama
- Qdrant vector database running (can be installed via Docker)
- Required Python packages from requirements.txt



### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Raane9401/LLM-Powered-Booking-Analytics-QA-System.git
cd hotel-booking-analytics
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Qdrant database**

I have used Qdrant Local which i got the image and ran container in local

```bash
docker run -p 6333:6333 qdrant/qdrant
```

5. **Place the dataset**

Place  hotel_bookings.csv file in the project root directory.

6. **Run data preprocessing and analytics generation**
```bash
python preprocess.py
```

This will:

- Clean the data
- Generate visualizations
- Store them in SQLite
- Create a cleaned CSV file

7. **Start the Flask API server**
```bash
python app.py
```

The server will start at http://localhost:5000

## Usage

### API Endpoints

1. **Get Analytics**
```bash
curl -X POST http://localhost:5000/analytics
```

Returns visualization metadata and summary statistics.

2. **Ask Questions**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the average revenue per booking?"}'
```

Returns an answer based on the data, along with performance metrics.

## Sample Test Queries \& Expected Answers

1. **Query**: "What is the average revenue per booking?"
**Expected Answer**: The average revenue is: 394.3079695867088
2. **Query**: "Which country has the most bookings?"
**Expected Answer**: Portugal (PRT) has the most bookings, followed by Great Britain (GBR), France (FRA), and Spain (ESP).
3. **Query**: "What is the cancellation rate for city hotels vs. resort hotels?"
**Expected Answer**: City hotels have a higher cancellation rate of approximately 41.7%, while resort hotels have a cancellation rate of about 27.8%. The overall cancellation rate across all hotels is 37.0%.
4. **Query**: "What is the the average cancellation rate?"
**Expected Answer**: The cancellation rate is: 27.489816467572886
5. **Query**: "Which market segment generates the most revenue?"
**Expected Answer**: The Online Travel Agency (TA/TO) market segment generates the most revenue, followed by Direct bookings. Corporate bookings, while fewer in number, tend to have higher average daily rates.

## Implementation Choices \& Challenges

### Architecture Decisions

1. **Data Processing Pipeline**
    - Implemented a robust preprocessing pipeline to handle missing values, inconsistent data types, and derived features
    - Created derived metrics like total stays and revenue to enable more comprehensive analysis
2. **Storage Strategy**
    - Used Qdrant for structured data and visualization metadata due to its simplicity and portability
    - Implemented Qdrant vector database for semantic search capabilities, enabling natural language queries
3. **API Design**
    - Built a RESTful API with Flask to provide a clean interface for accessing analytics
    - Included performance monitoring to track execution times and optimize slow queries

### Embedding \& LLM Integration

- Used HuggingFace embeddings (BAAI/bge-large-en) for vector representations i have used it for my hobby work
- Implemented fallback mechanisms between different LLM providers to ensure reliability
- Added context retrieval to provide relevant information to the LLM for more accurate answers


### Challenges \& Solutions

1. **Dependency Compatibility Issues**
    - **Challenge**: Encountered compatibility issues between Embeddings and ambiguity in answers lot of times
    - **Solution**: modified chunking with different ideas and sorting them
2. **Qdrant Integration**
    - **Challenge**: API changes in the Qdrant client and LangChain libraries caused initialization errors
    - **Solution**: Updated parameter names (using `embedding` instead of `embeddings`) and properly handled collection creation
3. **LLM API Reliability**
    - **Challenge**: Occasional timeouts and errors when calling external LLM APIs
    - **Solution**: Implemented a fallback mechanism that tries multiple LLM providers
4. **Performance Optimization**
    - **Challenge**: Slow response times for complex queries
    - **Solution**: Added caching for common queries and optimized vector search parameters
5. **Data Quality Issues**
    - **Challenge**: Missing values and inconsistent formatting in the dataset
    - **Solution**: Implemented robust data cleaning procedures and validation checks

## Future Improvements

1. Add user authentication and role-based access control
2. Implement caching for frequently accessed analytics
3. Add more advanced visualizations and interactive dashboards
4. Improve accuracy evaluation with more sophisticated metrics
5. Add support for real-time data ingestion and analysis


## Requirements.txt

```
Flask==3.1.0
flask_cors==5.0.1
langchain==0.3.21
langchain_community==0.3.20
langchain_core==0.3.47
langchain_huggingface==0.1.2
langchain_qdrant==0.2.0
matplotlib==3.10.1
numpy==1.24.1
openai==1.68.2
pandas==2.2.3
qdrant_client==1.13.3
Requests==2.32.3
seaborn==0.13.2

```

