# MicroBot

A simple chatbot site, based off some work I did for a university course at the university of Sydney. We had to code up and train a simple chatbot on the microsoft dialogues corpus (todo: add link), and evaluate it. The assignment involved testing and evaluating a few models, but I just include my final model (without any of the assignment writeup) here.

My model uses a simple Bahdanau attention mechanism with GRU layers. (TODO: add more details, maybe a diagram), and seems to work alright.

To run this locally, you'll need the data and word models I used, placed in a folder called 'data'. I'll release these publically at some point. After that, just install the required modules (using ```pip install -r requirements.txt```) and run

```
uvicorn main:app --reload --lifespan on
```

This starts the server, at port 8000 by default.
