from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from app.utils import process_file, make_prediction

import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Handle file upload, prediction, and rendering results."""
    try:
        contents = await file.read()
        df = process_file(contents)

        plot_path, forecast = make_prediction(df)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "forecast_table": forecast.to_html(classes="table table-striped"),
                "plot_path": plot_path,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})
