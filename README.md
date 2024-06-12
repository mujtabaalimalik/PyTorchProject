Created a neural network for image classification using convolutional layers. Here is the description of this project.

## Loading the Dataset
The dataset used is the CIFAR-10 dataset. A function was created to load the dataset. The function downloaded the dataset torchvision.datasets and loaded it using torch.utils.data.DataLoader. Both the test and train datasets were loaded using the above Python libraries.
Transformations for data augmentation were also applied in this function. This was done to improve the generalisation of the neural network. These transformations were only applied to the training dataset. Two transformations were used for data augmentation:
1.	A random horizontal flip using the torchvision.transforms.RandomHorizontalFlip.
2.	A random rotation between -5 and 5 degrees using torchvision.transforms.RandomRotation.
## Neural Network Architecture
### Intermediate Block
The `Block` class in the code implements our intermediate block. It only has convolutional layers and a fully connected layer at the end. Here are its important features:
* Convolutional layers: The block contains four convolutional layers. Each receives the same input image and outputs its own image. The parameters of each layer (kernel size, padding, and stride) are set so that all the layers output images of the same size.
*	Normalization and activation: Output of each layer is normalized and sent to a ReLU activation function.
*	Fully connected layer: There is one fully connected (linear) layer at the end to compute a vector $a$. A vector $m$ is computed in the forward function by computing the average value of each channel and storing into a variable `m`. This is then passed to the linear layer to compute the vector $a$. This layer has output units set to 4, which is the number of convolutional layers in our intermediate block.
*	Output formula: The vector $a$ (stored in variable `a`) has four elements, the same as the number of convolutional layers in our intermediate block. Each element of `a` is multiplied by the output of each of our layers and all the outputs of the convolutional layers are simply summed together. This gives us the out vector that we output for the block.
## Macroblock
The `Macroblock` class is added to produce cleaner code. It adds a level of abstraction so that we don't have to write a single long code for multiple blocks. This allows us to elegantly write how we wish to use our `Block` class. Here are its important features:
* Blocks: The macroblock has five blocks in succession: the output of the preceding block is the input of the next one. Here are the tuples of the input/output channels for our five blocks:
 * (3, 15)
 * (15, 30)
 * (30, 54)
 * (54, 75)
 * (75, 96)
*	We can see that the number of channels increase after each successive block. This is done intentionally to ensure that we have more capacity to continue to learn at each successive block.
## Output Block
The `OutBlock` class implements our output block. Here are its important features:
* Input: The output block's input is the output of our `Macroblock` (also the output of our last intermediate block, and the output of our last convolutional layer).
* Transformations: There are two things we do to the input before we send it to the fully connected layer:
 * We calculate the average of all channels using AdaptiveAvgPool2d. This applies a 2D pool over the input.
 * We reduce the number of dimensions by using Flatten.
* Fully connected layer: This linear layer has output channels set to the number we specified in the constructor. Later we will see that this number is 10 since that is required by us. The number of categories in our input data is 10.
* Output: The output block's output channels are 10.
## Model
The `Model` class implements model. It simply contains our `Macroblock` and `OutBlock` instances. Its constructor also accepts the number of outputs.
## Hyperparameters and Training
### Hyperparameters and Controls
Here is a list of the important hyperparameters that were used to train this neural network:
1.	Batch size: 128.
2.	Training epochs: 100.
3.	Learning rate: 0.01.
The initialization technique being used is the Xavier initialization. The optimizer being used is the Adamax optimizer. 
### Training and Testing
For every epoch:
1.	The model is trained, and its weights are updated.
2.	It is evaluated on the training and test dataset.
3.	The accuracies of the model on the train and test dataset are printed.
## Loss and Accuracies
The code also stores the loss for every batch, and train and test accuracies at every epoch. We have also plotted these for our best performing neural network. These can be seen below. The highest achieved on accuracy on the test dataset was 0.8759999871253967 (95th epoch).
<img width="209" alt="image" src="https://github.com/mujtabaalimalik/PyTorchProject/assets/157533823/40fc5816-3c96-4500-8e5a-cecf2df0523f">
<img width="217" alt="image" src="https://github.com/mujtabaalimalik/PyTorchProject/assets/157533823/793c1228-2552-4b6e-b12a-6fed4b593977">

## Improving Performance
Here is how we improved our performance. At first, we developed our network in pretty much the same way as explained. However, based on our experiments and findings, the model was improved to achieve higher accuracy.
### Changing Input and Output Channels for Blocks
At first, all our blocks had 3 input channels and 3 output channels. The model could not learn beyond 40% accuracy. Therefore, we changed the input/output channels of our blocks in a way that each block has more output channels than input channels. This dramatically improved our accuracies.
### Increasing the Number of Blocks and the Input/Output Channels
Initially, we only had three blocks. With the technique mentioned in the previous heading, there was a significant increase in accuracy but only as high as 70%. We experimented by adding additional blocks. We kept the number of input channels of the first block the same (3, the number of channels in our training data). But changed all other input/output channel values. Here is how we changed the blocks and their input/output channels:
1.	Block 1: (3, 9), Block 2: (9, 27), Block 3 (27, 81)
2.	Block 1: (3, 12), Block 2: (12, 48), Block 3 (48, 96) 
3.	Block 1: (3, 12), Block 2: (12, 33), Block 3 (33, 66), Block 4 (66, 96)
4.	Block 1: (3, 15), Block 2: (15, 30), Block 3 (30, 54), Block 4 (54, 75), Block 5 (75, 96)
5.	Block 1: (3, 12), Block 2: (12, 27), Block 3 (27, 42), Block 4 (42, 60), Block 5 (60, 75), Block 6 (75, 96)
Between the gain in testing accuracy was not that high when changing architectures 4 and 5 and the model became very complex since epoch times were very long. We decided to stay with the 4th architecture.
### Learning Rate
Initially learning rate was set at 0.1. This resulted in very erratic training patterns and missing of the minimal. The resulting graphs of losses and accuracies were also not smooth. The learning rate was reduced to 0.05, and then 0.01. The resulting graph was much smoother.
### Data Augmentation
We experimented with data augmentation techniques. At first, we only used the random horizontal flip, and it resulted in increased testing accuracy of the model with reduced overfitting. We decided to use another data augmentation technique: random rotation between -5 and 5 degrees. This also resulted in increased testing accuracy of the model with reduced overfitting. We stopped adding more since we crossed our target 85% accuracy.
