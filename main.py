from fastapi import FastAPI
from pydantic import BaseModel
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class Message(BaseModel):
    message: str

# chat api
@app.post("/chat")
async def send_message(message: Message):
    # we would put the prediction call in here
    return message.message

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

