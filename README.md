# ConvBot

The goal of this project is to create a chatbot that can sustain medium length
conversations and can build off of its own knowledge to learn new words that
are added to its vocabulary.

## Running
### Adding Training Data
To add data for the model to train on run:
```
python main.py --data_file=<data file>
```
This will start an infinite loop of getting your input and then prompting you
for a good response for that input. The contents will by written to <filename>
after they are parsed to make sure each word is in vocab.txt.

### Training
There are two ways to train: using a file containing training data or manually
entering a prompt and a good response for that prompt. To train using a data
file, setup the file as shown above and then run:
```
python main.py --train_file=<data file> --save=<save file> --train=all
```
You can train the generator, discriminator, or both models by setting the train
flag to gen, dis, or all respectively. You can also change the number of
training epochs before you are prompted to save with the save_itr flag.

To train the model by manually entering prompts and good responses, simply omit
the train_file flag. The load flag can be used to resume training from a saved
checkpoint:
```
python main.py --load=<saved file> --save=<save file> --train=all
```

### Chatting
Talking with the bot is done by omitting the train flag:
```
python main.py --load=<saved file>
```

## Expanding the Vocabulary
To add words to the vocabulary simply add each new word to then end of the
vocab.txt file. Currently, the model cannot transfer its learning to a model
with a larger vocabulary, so the model will need to be trained from scratch.
