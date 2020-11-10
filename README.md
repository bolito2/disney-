# WEED LMAO
### RNN trained to generate weed strain names

![Lmao](https://i.imgur.com/FqUL9PS.png)

Hey bro you got any of that Purice Kush, maybe some Masite Sipberry or Zonin Soops? Not even Purplos? I will have to settle with Entite OG

## I mean, why?
I've wanted to code a neural network from scratch to practice machine learning for a while, and because NLP is one of the coolest fields right now I decided to make a name generator. I know that given that it's a simple model(plain RNN) it will produce mostly gibberish so what better dataset to train it that with weed strain names 👌😂 

Thanks to [LiamLarsen](https://www.kaggle.com/kingburrito666/followers) for offering the dataset over at [Kaggle](https://www.kaggle.com/kingburrito666/cannabis-strains)

## How to use it
### Installing
To make your own weed names just open bash/powershell, clone the repository and install the dependencies with
```bash
git clone https://github.com/bolito2/weed-lmao
cd weed-lmao
pip install -r requirements.txt
```
They are only numpy for the linear algebra, matplotlib for plotting the cost and h5py for saving and loading the parameters. You may be asking; why didn't you use Tensorflow haha what a fucking tryhard. Well, I'm not a CIA n*****, I write my own neural networks. 

### Generating names
If you want to get straight to the action, the repository contains the pre-trained parameters(yeah, I know) so just type
```bash
python weed_lmao.py generate
```
It will open a prompt for you to write how the name should start. If you leave it blank it will start with a random letter. When you want to exit, enter a backslash(\)

Most of the outputs aren't very interesting, but sometimes you get names that seem what GPT-3 or other reputable algorithm would output if it had consumed the strain that it was trying to name, so pretty good results overall for what i was expecting.

### Custom training
If you want you can also train the network at your taste. Keep in mind that I don't work at Google so this may take some time to train even though it is a pretty shitty model. To train it from scratch you have to delete the file parameters.h5, or else it will keep training over those parameters. The command is the following:
```bash
python weed_lmao.py train
```
You can choose these options:
* -u -> number of units of the RNN(I will explain it below)
* -r -> learning rate
* -e -> number of epochs

The default command above is equivalent to
```bash
python weed_lmao.py train -u 10 -r 0.05 -e 50
```

I haven't really put much thought in the defaults, if you make some tries you will probably come with a better combination

