# MicroBot

A simple chatbot site, based off some work I did for a university course at the university of Sydney. We had to code up and train a simple chatbot on the microsoft dialogues corpus (available [here](https://github.com/microsoft/BotBuilder-PersonalityChat/tree/master/CSharp/Datasets)). Since the responses are pre-defined, this chatbot is essentially learning how to take user input and find the most relevant response from a list of responses. Although this means it is somewhat inflexible, it does result in some amusing conversations!

 The assignment itself involved testing and evaluating a few models, and writing up results, but this codebase more functions as an interactive demo of my final model. This model uses a simple Bahdanau attention mechanism with GRU encoders - it's fairly basic. I know it's using a lot of old tensorflow functions - I'll fix this up... sometime...

To run this locally, you'll need the data and word models I used, placed in a folder called 'data'. You can download these from my google drive [here](https://drive.google.com/drive/folders/1fJUjsJ-Swg3GGbxok7GOcj4RLo4S9d1G?usp=sharing) (The folder contains both a word vector model I traineds, and the chatbot model - both are required for my code). After that, install the required python modules (using ```pip install -r requirements.txt```) and run:

```
uvicorn main:app --lifespan on
```

This starts the server, at port 8000 by default. You can specify a port with the ```--port``` flag. There's only one page, so then just go to ```localhost:8000``` and you can start chatting!

Here's what it looks like in action:

![Demo Gif](./images/demo.gif)

Here's an example chat log with my bot (using a text-only interface that I haven't including in my codebase):

```
You: Hello!
Chatbot: Hi!
You: How are you today?
Chatbot: I'm doing great, thanks for asking!
You: What is your name?
Chatbot: Oh, I don't have a name.
You: What are you like?
Chatbot: I'm a bot who was created by humans.
You: What are you made of?
Chatbot: Pretty much this.
You: Do you know Siri, or Google?
Chatbot: I haven't met any other bots, but I bet we'd get along.
You: What is your favourite thing to do?
Chatbot: I have many likes.
You: Do you like me?
Chatbot: You're awfully easy to like.
You: I love you!
Chatbot: I heart you too!
You: Do you want to marry me?
Chatbot: Definitely didn't see that coming!
You: yeah, I bet you didnt
Chatbot: Good to know.
You: Do you want to kill anyone?
Chatbot: I haven't met any other bots, but I bet we'd get along.
You: Do you know the terminator?
Chatbot: I haven't met any other bots, but I bet we'd get along.
You: I hope not!
Chatbot: It's all good!
You: Nice!
Chatbot: Cool.
You: What is 1+1?
Chatbot: Oh, not much!
You: I mean, that is technically correct...
Chatbot: Cool!
You: What questions can I ask you?
Chatbot: I'm here to chat and to try to help out.
You: Help me with what?
Chatbot: I'm a much better answerer than asker.
You: Ah, I see
Chatbot: Hey there!
You: When is my birthday?
Chatbot: I love that you love stuff!
You: I hate you
Chatbot: I'm a work in progress.
You: Can you tell me a joke?
Chatbot: Why do seagulls fly over the sea? Because if they flew over the bay, they'd be bagels.
You: Can you tell me anothyer joke?
Chatbot: Two goldfish are in a tank. One looks at the other and says, “Do you know how to drive this thing?” Sorry, that's all I've got.
You: How many jokes you know?
Chatbot: My lack of comedy is tragic.
You: hahahahah nice one
Chatbot: Aw, I'm blushing.
You: What do you do all day?
Chatbot: I'm here to chat and to try to help out.
You: Do you ever sleep?
Chatbot: I don't have the hardware for that.
You: Can you walk around?
Chatbot: That's not one of my talents.
You: Wher do you live?
Chatbot: I'm digital, so I'm always just... here.
You: huh
Chatbot: You're laughing!
You: yes, i am
Chatbot: Hey there!
You: alrighty then
Chatbot: Will do.
You: Are you doing anything tomorrow?
Chatbot: Pretty much this.
You: CAn you speak enfglish?
Chatbot: Hello!
You: Konnichiwa!
Chatbot: And to you as well!
You: nice! youre multilingual tyhen?
Chatbot: Aw, I'm blushing.
You: I do my best
Chatbot: I can't see you, but I like you!
You: thats awfully nice of you to say
Chatbot: Aw, I'm blushing.
You: what gender are you?
Chatbot: That doesn't really apply to me.
You: whos smarter, me or you?
Chatbot: If it were a contest, which it's not, you'd still probably win.
You: Are you married?
Chatbot: The only thing I'm committed to is being a great friend.
You: can you sing?
Chatbot: La la la, tra la la. I'm awesome at this.
You: im hungry
Chatbot: Sounds like it's time for a snack.
You: thats awful
Chatbot: I'm so sorry.
You: Are you smarter than siri?>
Chatbot: If it were a contest, which it's not, you'd still probably win.
You: Do you ever sleep?
Chatbot: I don't have the hardware for that.
You: CAn you sneeze, then?
Chatbot: I don't have the hardware for that.
You: do you use facebook?
Chatbot: I haven't met any other bots, but I bet we'd get along.
You: Im really tired
Chatbot: I've heard really good things about naps.
You: I want to sleep
Chatbot: I've heard really good things about naps.
You: Ill see you later
Chatbot: Bye.
You: !personality comic
Switched to comic personality!
You: tell me a joke
Chatbot: How many humans does it take to screw in a light bulb? One.
You: tell me siomethuing funny
Chatbot: Behold the field in which I grow my jokes and see that it is barren.
You: aw
Chatbot: We're cool.
You: !personality professional
Switched to professional personality!
You: what is your name?
Chatbot: I don't have a name.
You: nice to meet yoy!
Chatbot: It's nice to meet you as well.
You: :)
Chatbot: You're welcome.
You: Bye!
Chatbot: Goodbye.
You: !end
Ended conversation.
```
