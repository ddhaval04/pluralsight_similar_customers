from flask import Flask, jsonify, Response, request

from pluralsight_similar_customers.models.customer_similarity_model import (
    CustomerSimilarityModel,
)

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to the Pluralsight MLE Tech Challenge"


@app.route("/predict", methods=["POST"])
def predict_model():
    user_handle = request.get_json()["user_handle"]
    customer_similarity_model = CustomerSimilarityModel()
    customer_similarity_model.load_models_from_files()
    try:
        idx = customer_similarity_model.index_metadata["customer_list"].index(
            user_handle
        )
        query_embedding = customer_similarity_model.index_metadata[
            "customer_embeddings"
        ][idx]
        query_embedding = query_embedding.reshape(1, -1)
        predictions = customer_similarity_model.predict(query_embedding)

        return jsonify(
            {
                "user_handle": predictions["user_handle"].tolist(),
                "viewed_courses": predictions["course_tags"].tolist(),
                "course_viewe_time_seconds": predictions["view_time_seconds"].tolist(),
            }
        )

    except ValueError as error:
        return Response(
            json.dumps("User handle not present in the training data."), status=400
        )
