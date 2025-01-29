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
        df = preprocess_data(df)

        model = train_model(df)

        predictions = predict_weather(model, df)

        df['predictions'] = predictions

        return templates.TemplateResponse("index.html", {"request": request, "predictions": df.to_html(classes='mystyle')})

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)