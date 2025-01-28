from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
from app.models.prophet_model import prophet_forecast
from app.models.sarima_model import sarima_forecast

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    return templates.TemplateResponse("upload.html", {"request": {}})

@app.post("/forecast")
async def forecast(file: UploadFile, model: str = Form(...), periods: int = Form(...)):
    # Save the uploaded file
    file_path = Path("data") / file.filename
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Load the data
    data = pd.read_csv(file_path)

    # Validate data columns
    if not {"ds", "y"}.issubset(data.columns):
        return {"error": "Invalid file format. Ensure columns 'ds' and 'y' are present."}

    # Forecast based on selected model
    if model == "prophet":
        forecast_df = prophet_forecast(data, periods)
    elif model == "sarima":
        forecast_df = sarima_forecast(data, periods)
    else:
        return {"error": "Invalid model selection."}

    # Debugging: Print forecast data to ensure it's being passed correctly
    print(forecast_df.head())  # Check the forecast data

    # Render the results in the HTML template
    return templates.TemplateResponse("results.html", {"request": {}, "data": forecast_df.to_dict(orient="records")})

