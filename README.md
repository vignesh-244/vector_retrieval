# Vector Retrieval

This project is a small Flask app that searches a folder of text documents and returns the most relevant matches for a query.

I built the search logic from scratch instead of using a ready-made NLP package. The app reads `.txt` files from the `documents/` folder, converts them into TF-IDF vectors, and ranks matches with cosine similarity.

## What the project does

- loads text files from the local `documents/` folder
- tokenizes and indexes the content
- accepts a search query through an API or simple browser page
- returns the top matching documents with a short snippet

## Project files

```text
.
├── app.py
├── search_engine.py
├── documents/
├── templates/
└── requirements.txt
```

## How to run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

After that, open:

```text
http://127.0.0.1:5000
```

## API routes

### Search

```http
GET /search?q=artificial intelligence in finance
```

Example response:

```json
{
  "query": "artificial intelligence in finance",
  "results": [
    {
      "document": "finance_ai.txt",
      "score": 0.4721,
      "snippet": "Artificial intelligence is changing the finance industry in practical ways. Banks use machine learning systems to detect fraud patterns and reduce risk."
    }
  ]
}
```

### Rebuild the index

```http
POST /index
```

Example response:

```json
{
  "message": "Index rebuilt successfully",
  "documents_indexed": 5,
  "vocabulary_size": 114
}
```

## How the search works

1. The app reads every `.txt` file inside `documents/`.
2. Each file is converted into lowercase tokens using regex.
3. TF-IDF scores are calculated for the document terms.
4. The query is converted into the same vector format.
5. Cosine similarity is used to rank the closest matches.
6. The top results are returned as JSON.

## Notes

- The sample text files can be replaced with any other `.txt` dataset.
- If you add or remove files, call `POST /index` to rebuild the search index.
- The UI on `/` is only for basic testing; the main logic is in the backend.
