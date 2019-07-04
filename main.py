from fastapi import FastAPI
from pydantic import BaseModel
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from chatbot import load_chatbot, get_response

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

items = {}

# load our model on startup
@app.on_event("startup")
def load_model():
    items['sess'], items['wv_model'], items['models'], items['answer_sets'] = load_chatbot()

class Message(BaseModel):
    message: str
    mode: str

# the chatbot has no coreferencing or anything,
# so we can treat each response individually on its own.
@app.post("/chat")
async def send_message(message: Message):
    return get_response(items['sess'], items['wv_model'], items['models'], items['answer_sets'], message.message, message.mode)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

