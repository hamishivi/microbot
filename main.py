import os
from fastapi import FastAPI, HTTPException
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

# the chatbot has no coreferencing or notion of memory,
# so we can treat each user input individually without saving
# user details.
@app.post("/chat")
async def send_message(message: Message):
  if 'sess' not in items or 'wv_model' not in items or 'models' not in items or 'answer_sets' not in items:
    raise HTTPException(status_code=500, detail="Server Error - chatbot couldn't load!")
  if not (message.mode == 'friend' or message.mode == 'professional' or message.mode == 'comic'):
    raise HTTPException(status_code=400, detail="Mode can only be 'friend', 'professional', or 'comic'.")
  return {"answer": get_response(items['sess'], items['wv_model'], items['models'], items['answer_sets'], message.message, message.mode)}

@app.get("/")
async def root(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})

