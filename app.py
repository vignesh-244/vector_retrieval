from flask import Flask, jsonify, render_template, request

from search_engine import SearchEngine


app = Flask(__name__)
engine = SearchEngine(documents_dir="documents")
engine.build_index()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    results = engine.search(query)
    return jsonify({"query": query, "results": results})


@app.route("/index", methods=["POST"])
def rebuild_index():
    engine.build_index()
    return jsonify(
        {
            "message": "Index rebuilt successfully",
            "documents_indexed": len(engine.documents),
            "vocabulary_size": len(engine.vocabulary),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
