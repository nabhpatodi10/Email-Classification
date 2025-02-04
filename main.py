import pandas as pd
import joblib
from fastapi import FastAPI
import uvicorn

vectorizer = joblib.load("vectorizer")

gnb = joblib.load("GaussianNB")
lrc = joblib.load("LogisticRegression")
mnb = joblib.load("MultinomialNB")
rfc = joblib.load("RandomForest")

app = FastAPI()

@app.get("/")
def home():
    return {"Home" : "Welcome User!"}

@app.post("/classify")
def classify(mail: str):
    data = pd.DataFrame({"text" : [mail]})

    inp = vectorizer.transform(data["text"]).toarray()

    gnb_prediction = gnb.predict(inp)
    lrc_prediction = lrc.predict(inp)
    mnb_prediction = mnb.predict(inp)
    rfc_prediction = rfc.predict(inp)

    for i in range(len(gnb_prediction)):
        if gnb_prediction[i] + lrc_prediction[i] + mnb_prediction[i] + rfc_prediction[i] > 1:
            return "Spam"
        else:
            return "Not Spam"
        
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)