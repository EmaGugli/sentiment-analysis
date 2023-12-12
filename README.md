# Sentiment Analysis
This project focuses on sentiment analysis using a logistic regression model trained on Danish sentences. The model is built to predict sentiment labels (positive, negative, neutral) based on input text, allowing for an understanding of the emotional tone or sentiment expressed in the provided text.
It was used the multilingual "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" model to encode the text.

## Project Structure
The project is organized as follows:
```markdown
├── data
│ ├── train.parquet
│ └── test.parquet
├── models
│ ├── label_encoder.joblib
│ └── logistic_regression_model.joblib
├── src
│ ├── train.py
├── Dockerfile
├── main.py
├── README.md
└── requirements.txt
```
data: Directory containing training and testing data in Parquet format. Downloaded from https://huggingface.co/datasets/DDSC/angry-tweets

requirements.txt: File specifying the Python dependencies required for running the project.

models: Directory containing the label encoder (label_encoder.joblib) and logistic regression model (logistic_regression_model.joblib) used for sentiment analysis.

src/train.py: Script for training the logistic regression model. The script uses the SentenceTransformer library to embed text and trains a logistic regression model on the labeled data. The trained model and label encoder are saved in the models directory.

Dockerfile: Docker configuration file for containerizing the project and serving it through fastAPI. It uses the model files and the main.py script.

main.py: FastAPI application defining an API endpoint for sentiment analysis. The API takes input text and returns the predicted sentiment label using the trained logistic regression model.

## Usage
```bash
git clone <repository-url>
cd <repository-directory>
docker build -t sentiment-analysis-app .
docker run -p 80:80 sentiment-analysis-app
```

Access the API at http://localhost:80/docs to interact with the sentiment analysis endpoint.

## API Endpoint
POST /sentiment_analysis: Endpoint for predicting sentiment. Send a POST request with JSON data containing the text to analyze. The response includes the predicted sentiment label.

Payload/response example:
```bash
Payload:
{
  "text": "This is a positive example."
}

Response:
{
  "prediction": "positive"
}
```

CURL example
```bash
curl -X POST "http://localhost:80/sentiment_analysis" -H "accept: application/json" -H "Content-Type: application/json" -d '{"text": "This is a positive example."}'
```