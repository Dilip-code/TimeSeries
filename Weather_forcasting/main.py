from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from model import train_model, predict_weather
from utils import preprocess_data

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        # Preprocess the data (utils.py)
        df = preprocess_data(df)

        # Train the model (model.py)
        model = train_model(df)

        # Make predictions (model.py)
        predictions = predict_weather(model, df)

        # Add predictions to the DataFrame
        df['predictions'] = predictions

        # Render the predictions in the template
        return templates.TemplateResponse("index.html", {"request": request, "predictions": df.to_html(classes='mystyle')})

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)