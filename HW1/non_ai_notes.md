## Preface

***In this markdown I just plan to write my thought process/approach to all these parts in the HW. This is my Non-AI attempt, and my undertsanding, and approach to how I want to solve the problem and then bring it to fruition with the help of AI. The professor mentioned this in class and I think this would be helpful for grading. I will be using AI for every part of this HW, just to make my submission cleaner and also for accuracy!***


## Part A

### Overview / Thought Process 

In class we talked about different activation functions, and in chapter 6.3 the notes talked about activation functions. So for my Part A I just wnat to demonstrate results of a neural network trained on the same paramters but the only difference being the the activation functions.

I want to show the differnece between Sigmoid and Step activation functions. We discuss the cons of a hard threshold of the step function and how the sigmoid was a bettwer option but adding a visualization and ocmparing the results would be nice.

For them demo I want to train a model on my two favorite pokemon Snorlax and Mudkip, and then just show the training and validation reults.

Also at the bottom Ill include an imgae upload where you can uplaod and image and see how each model does a predicting wether or not one is snorlax or mudkip to make it interactive.

### Image Preprocessing 

I know we have to normalize all the images and fit them to the same dimensions to pass through the model:
 - fit to a certain dimension so DxD, 64 by 64 i forget the correct one or 128x128
 - will most likely need to take away the color and greyscale the images because we are focused more on the shape of the pokemon and not the color, and there are shiny versions of each

### Model Params

There are the general fixed params I'll have for each model, I rmemebr from NLP we had to to an image classifier and these are some of the params I rememebr using:

 - 3 Hidden Layers
 - learning rate: 0.001
 - Epoch: 30 
 - Activation Function: Sigmoid or Step

### Gradio layout

General formulas and explanation at the top about Step vs Sigmoid, with fixed params

A button to train both models Simultaneously and then charts to compare:
 - Training vs Validation Accuracy 
 - Confusion Matrix to show results 

At the bottom image uplaod and the results from each model and their outputs.


## Part B.0

I didn't read ahead but i already included my favorite character which is Snorlax in Part 1 lol. 

But I think for this one I want to kinda demonstrate the pre image processing that is needed to be done to train neural networks as image classifiers:

 - Show original image
 - Show pixelated verison
 - Show pixelatd greyscale version
 - Maybe show how neural networks read these vectors or flattened images when we reduce the size.
 - I think it goes on the darkness of each pixel and produces a value.
    - confirmed with AI its called Intensity the variation in color of a pixel from white to black, so graphs that demonstrate this



## Part B.1

### Finding the Error

While interacting with the params in the perception lab notebook. I saw when i left everything deafult and moved the bias to the max or min side, it seemed ike the data disappaard from the grapgh, and what orighinally where wasnt.

he graphs where also bounded by x=10 and y=10, so this led me to thing we werent visulaly seeing all the data.

I then asked AI if this could be the issue and it explain yes that the visualizations were bounded by 10 and that the boundary line was also bounded by 10.


### Possible Solution

AI gave me a hint for possible solutions already so this isnt super original, but fitting the bounds of the graph to the min and max data points in the dataset. So the range will be dynamic/ adaptive to the outputs and eveyrthing will be displayed properly then.

### Gradio Layout
 
For the layout i think just putting two graphs side by side to compare will be the best option. So I'll just make dummy data and show a graph with the boundary constraint and then the proper graph with a dynamic boundary where you can see all the data.

## Part B.2

I'm just gonna go through and answer the questions and then display it on gradio somehow, ill probably include my thooughts there.

## Part B.3

I decided to take Data and Modeling, the main reason was that the instructions were shorter lol. But I also liek using kaglle datasets so I had some familiarity with this, and predicting death based on stats is interesting to me.

### Structure and Meaning of Data
Im looking at the columns and trying to undertsand what data should be weighted more in terms of if someone would die or not:

 - id: The unique identifier for each character, because even in the Star Wars universe, we need order.
    - Not needed
 - name: From Anakin to Zuckuss, it’s the name that makes the legend.
    - I would say not needed, I dont think a specific name would be valuable, and not sure how it would even get interpreted through the model
 - species: Whether they’re human, Wookiee, or Ewok, find out who belongs to what
 species.
    - I think this is valuable, because droids are probably more likely to die, and there probably is some statistical significane for this column
 - gender: Male, female, droid, or otherwise, everyone’s included.
    - This is vlauable also, similar reaosn to the species column
 - height: Measured in meters, because Star Wars doesn't do inches.
    - I would say yes, if your tall more body to be shot or sliced by a sword is my assumption
 - weight: In kilograms – blame the galaxy's standard metric system.
    - A similar reason for height 
 - hair_color: From luscious locks to shiny domes, it’s all here.
    - we cna have this but maybe with a lower weight, maybe hair or baldness can add camofaluge lol 
 - eye_color: You can gaze into the yellow eyes of the Sith safely.
    - Maybe people with certain eye colors, have better survival skills, so i would say a medium weight 
 - skin_color: Including shades like “pale Sith” and “green Yoda.”
    - I would also say this is valuable, medium weight, certain skin colors may have a better survival rate
 - year_born: BBY (Before the Battle of Yavin), because who needs BCE?
    - I would say yes, older charcters may be more likely to die, high weight
 - homeworld: Where they come from, like Tatooine or Alderaan (before it went boom).
    - Valuable, I think the deathstar destroyed certain worlds and maybe some worlds dont have good training
 - year_died: Not all heroes live forever, but they all get a date (or a blank if they're lucky).
    - I think this will be used to validate the data, so if predicted yes died then we see if not null and in vice versa
 - description: Brief bios, because everyone deserves a little backstory.
    - I think this would require some nlp or something so i would sya not to include it tbh

### Linear Regression and Neural Networks

Plan on comparing linear regression vs the neural network and compare these results

### Modeling Choices

Would use the binary cross entorphy we talked about in class, add some charts about loss and accuracy

Also use Adam model which i think is the best for learninhg, i forget the specific name its used for but its called adam 

And then also any charts that are important for evaluation. That AI says will be good, Im not too sure what evaluation metrics will be good besides accuracy and the trining loss.

### Interactive

I think adding an area where you can make up a character and suybmit it and see if they die based on the params would be interesting so ill add that. I feel like most people die in star wars tho, or how do you classify like yodas ghost is he living in spririt or technically dead?? But regardless I think this will be fun.

I put my metrics in tehre and said i was from mandalore and I died :(

## Part C

Im not sure what to do for this form will ask in class. NVM I read the rest of it

## Part C.A


