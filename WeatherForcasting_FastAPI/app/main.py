from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from app.utils import process_file, make_prediction

import os

# Initialize FastAPI app
app = FastAPI()

# Mount static files for CSS
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates for HTML
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Handle file upload, prediction, and rendering results."""
    try:
        # Read file and process
        contents = await file.read()
        df = process_file(contents)

        # Generate prediction and plot
        plot_path, forecast = make_prediction(df)

        # Render result template
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
