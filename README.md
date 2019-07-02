# MicroBot

A simple chatbot site, based off some work I did for a university course at the university of Sydney. We had to code up and train a simple chatbot on the microsoft dialogues corpus (todo: add link), and evaluate it. The assignment involved testing and evaluating a few models, but I just include my final model (without any of the assignment writeup) here.

My model uses a simple Bahdanau attention mechanism with GRU layers. (TODO: add more details, maybe a diagram), and seems to work alright.

I'm serving the api and website with FastApi, and hosting with heroku - check it out here (todo: add heroku details). Note that I'm using the heroku free tier, so it will take a while to startup and will be very slow to run.
