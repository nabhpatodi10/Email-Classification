# Email-Classification

This repository contains a machine learning-based email classification project. The application uses multiple machine learning models to classify emails as **Spam** or **Not Spam** through a REST API built using FastAPI.

## Features

- **Multi-Model Classification:** Uses **Gaussian Naïve Bayes**, **Logistic Regression**, **Multinomial Naïve Bayes**, and **Random Forest** to classify emails.
- **FastAPI Integration:** Provides a simple REST API for email classification.
- **Vectorization of Email Texts:** Uses a trained vectorizer to transform email text into a numerical format for classification.
- **Simple API Structure:** Accepts email text via a POST request and returns classification results.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.10 or above
- `pip` package manager

### Setting Up the Environment

1. **Clone the Repository**
   ```sh
   git clone https://github.com/your-username/email-classification.git
   cd email-classification
   ```

2. **Create a Virtual Environment (Optional)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Ensure the following Python libraries are installed:
  
   ```
   fastapi
   uvicorn
   pandas
   joblib
   scikit-learn
   ```
  
   Install them using:
    ```sh
    pip install scikit-learn pandas fastapi uvicorn
    ```

5. **Ensure the Trained Models Exist**
   The API expects the following trained models and vectorizer to be present in the root directory:

   - `vectorizer` (Trained text vectorizer)
   - `GaussianNB` (Gaussian Naïve Bayes Model)
   - `LogisticRegression` (Logistic Regression Model)
   - `MultinomialNB` (Multinomial Naïve Bayes Model)
   - `RandomForest` (Random Forest Model)

   If these models are not present, you will need to train and save them.

## Running the API

Run the FastAPI application using `uvicorn`:

```sh
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

After starting the server, the API will be accessible at:

- **Home Route:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Docs (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Endpoints

### 1. **Home Route**
   - **Endpoint:** `/`
   - **Method:** `GET`
   - **Response:** `{ "Home": "Welcome User!" }`

### 2. **Classify Email**
   - **Endpoint:** `/classify`
   - **Method:** `POST`
   - **Request Body:** 
     ```json
     {
       "mail": "Your email content here"
     }
     ```
   - **Response:** `"Spam"` or `"Not Spam"`

## Example Usage

You can test the classification endpoint using **cURL**:

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/classify' \
  -H 'Content-Type: application/json' \
  -d '{"mail": "Congratulations! You have won a lottery."}'
```

Or using Python:

```python
import requests

url = "http://127.0.0.1:8000/classify"
data = {"mail": "Congratulations! You have won a lottery."}

response = requests.post(url, json=data)
print(response.text)
```

## File Structure

```
📂 email-classification
│── main.py               # FastAPI app for email classification
│── notebook.ipynb        # Jupyter Notebook for model training and evaluation
│── vectorizer            # Saved text vectorizer (required)
│── GaussianNB            # Saved Gaussian Naïve Bayes model
│── LogisticRegression    # Saved Logistic Regression model
│── MultinomialNB         # Saved Multinomial Naïve Bayes model
│── RandomForest         # Saved Random Forest model
│── README.md             # Project documentation (this file)
```

## Model Training

The **`notebook.ipynb`** file contains code for training the models. If you need to retrain, follow the steps in the notebook to:

1. Load and preprocess the dataset.
2. Train the models.
3. Save the trained models and vectorizer using `joblib`.

## Future Enhancements

- Improve accuracy using deep learning-based models (LSTM, BERT).
- Deploy the API using Docker and cloud services.
- Implement a frontend UI for user-friendly classification.
- Enhance email preprocessing to improve classification performance.

## License

This project is open-source under the **MIT License**.
