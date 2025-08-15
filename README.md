# Hybrid Text-to-Video Search System

## How It Works

1.  **Data Ingestion**:
    -   Video metadata (title, description, etc.) from `.json` files is indexed into an Elasticsearch index.
    -   Pre-computed CLIP feature vectors (`.npy` files) for video keyframes are inserted into a Milvus collection.
2.  **Search Execution**:
    -   A user's text query is sent directly to Elasticsearch for keyword matching.
    -   The same text query is encoded into a vector by the CLIP model and used to find semantically similar keyframes in Milvus.
3.  **Result Fusion**:
    -   The ranked lists of videos from Elasticsearch and Milvus are combined using RRF. This produces a final, unified ranking that leverages both keyword relevance and semantic context.

## Setup and Usage

### 1. Prerequisites

-   Docker and Docker Compose
-   Python 3.8+
-   Your video metadata (`.json`) and CLIP feature vectors (`.npy`)

### 2. Configure the Environment

Create a `config.py` file and populate it with the necessary paths and settings.

### 3. Start Services

```bash
docker compose up -d
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the System

```bash
streamlit run app.py
```