# Disney-
### Recurrent neural network trained to generate movie titles

Hey you wanna watch Lagrini? Or maybe the new series Lie and Okal, I heard it's good. I'm not in the mood to watch Wire of Coogulim right now.

## Why?
I've wanted to code a neural network from scratch to practice machine learning for a while, and because NLP is one of the coolest fields right now I decided to make a name generator. I know that given that it's a simple model(plain RNN) it will produce mostly gibberish so what better dataset to train it than with movie titles.

Thanks to [Yash Goes]([https://www.kaggle.com/kingburrito666](https://www.kaggle.com/goelyash)) for offering the dataset over at [Kaggle]([https://www.kaggle.com/kingburrito666/cannabis-strains](https://www.kaggle.com/datasets/goelyash/disney-hotstar-tv-and-movie-catalog))

## How to use it
### Installing
To get ideas for your own movies just open bash/powershell, clone the repository and install the dependencies with
```bash
git clone https://github.com/bolito2/disney-
cd disney-
pip install numpy matplotlib
```
They are only numpy for the linear algebra and matplotlib for plotting the cost. You may be asking; why didn't you use Tensorflow?? xd. Well, I wanted to write my own neural network. 

### Generating titles
If you want to get straight to the action, the repository contains the pre-trained parameters(yeah, I know) so just type
```bash
python disney-.py generate
```
It will open a prompt for you to write how the name should start. If you leave it blank it will start with a random letter. When you want to exit, enter a backslash( \\ )

### Custom training
If you want you can also train the network at your taste. Keep in mind that I haven't optimized it at all so this may take some time to train even though it is a pretty basic model. To train it from scratch you have to delete the file parameters.npz, or else it will keep training over those parameters. The command is the following:
```bash
python disney-.py train
```
You can choose these options:
* -u -> number of units of the RNN(I will explain it below)
* -r -> learning rate
* -e -> number of epochs

The default command above is equivalent to
```bash
python disney-.py train -u 32 -r 0.07 -e 100
```

I haven't really put much thought in the defaults, if you make some tries you will probably come with a better combination.

## How does it work?
### Training
At its core this project is a Recurrent Neural Network(RNN), which work pretty well for data that has a sequential component(like letters in a name) for its complexity. Here is a picture of the model I used, when trained with the name 'gravity':

![Training](https://i.imgur.com/AH9s8yK.png)

The elements in the picture are **x**, the inputs, column vectors of length *n_letters*(the number of supported characters, explained below), **a**, the hidden states of length *units*, **y** the intermediate outputs of length *n_letters* and **p**, the final predictions(probability of next letter being that index) of length *n_letters* too.

Then there's the parameters, **Wx**, **Wa**, **Wy** and the biases **ba**, **by** not in the picture, which are used to compute all those things. One important fact is that they are shared along all time-steps. It really is a single RNN cell feeding its outputs into itself over and over. This is what makes RNN light-weight and flexible for variable-length inputs like strings of letters.


#### Forward propagation
In each time-step(left to right) we start with the letter of the word corresponding to that time-step(**x**) and encode in a one hot way. Basically we use an array containing all the letters supported by the algorithm, in this case ' abcdefghijklmnopqrstuvwxyz>' that is, the letters a-z, spaces and >, a special character I will explain shortly. Then, create a vector where all its entries are zeros except the one in the index of the letter in the previous array(That's why it's called *one-hot* encoding).

Then we have the hidden state of the RNN(**a**) which is what keeps track of the previous letters to find patterns, so *units* can be seen as the size of the RNNs "memory". This is combined with the input using the weights **Wx** and **Wa** to get the next hidden state, which get passed directly to the next time-step and is also used to compute the intermediate output **y**. I used the hyperbolic tangent as activation function.

Then that goes through a softmax layer so all its entries are positive and sum to one(they will be used as probabilities in generation) and this prediction is compared to what the next letter really is to get the cost. That continues until the last time-step, which is always trying to predict the end token, >. The reason for using this token(that gets added when loading the database) will be explained in the **Generating names** section.

Then we sum the costs computed at each time-step with the cross-entropy formula and get a value for how far-off the predictions were.

#### Back-propagation through time
Then apply backpropagation through time (not going explain that here lol) to get the gradient of the cost with respect to each parameter(**Wx**, **Wa**, **Wy** and the biases **ba**, **by** not in the picture) and use gradient descent to hopefully get better predictions next time. 

I actually backpropagate all the way to the start although the gradients quickly decrease to zero so I don't think it is really worth it for the extra computation time but whatever. A good solution for this problem(vanishing gradients) would be switching the plain RNN with a LSTM but I'm not going to go through that to generate movie titles, sorry. If you are interested in getting better results and don't care about how this things work under the hood I suggest you to use Tensorflow or other machine learning library.

This cycle of forward-propagation and back-propagation is computed with each word of the dataset in every epoch, for as many epochs as you input.

### Generating names
Okay, we(well, our CPUs) had to GRIND to get here, but the struggle was for something. Now with a few tweaks the RNN can start to generate its own names.
![Generating](https://i.imgur.com/CETqdpE.png)

That's pretty cool, isn't it? What we are doing is feeding the input('owo' in this example) to the network like in training, but discarding the output until the last letter. This is because even thought we don't use the output anywhere, the hidden values will reflect the whole input so the RNN will be able to predict patterns that occurred in the input string. 

Then, the interesting part. From the last letter of the input, we take the predictions **p3** and sample a character using it as the probability of choosing each one(**p** is a vector of length *n_letters* and the softmax layer has normalized it). We save that character to the generated name and feed it to the input of the next time-step, as a one_hot vector, not the probabilities. Then, this process is repeated for as long as the network wants, until it chooses the end token(>) as the next letter, where the process will stop and the generated word can be returned.

And that's everything, have fun with it!
