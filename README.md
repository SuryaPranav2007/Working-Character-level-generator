# Working-Character-level-generator

Pipeline Architecture

The Makemore project implements a character-level language model that learns character-transition patterns from a dataset of 32,033 names and generates new name-like sequences.

1. Data Loading & Preprocessing

Load the names.txt dataset containing 32,033 names.
Extract individual characters and introduce . as a special token representing the beginning and end of a name.
Construct input-target pairs for next-character prediction.

2. Character Encoding

Map each character to a unique integer using stoi and itos dictionaries.
Apply one-hot encoding in the neural bigram model to represent characters as categorical vectors.
In the MLP model, use a learnable embedding matrix to transform characters into dense numerical representations.

3. Bigram Model

Construct a 27 × 27 frequency matrix representing consecutive character pairs.
Normalize character-transition counts into probability distributions.
Generate names by sampling the next character based on the current character's transition probabilities.

4. Neural Network Architecture

Replace the explicit bigram lookup table with a trainable neural network.
Compute logits representing unnormalized scores for possible next characters.
Apply the softmax function to convert logits into probability distributions.

Model Training & Optimization

Train the model using mini-batches of 32 examples.
Apply backpropagation and automatic differentiation to compute parameter gradients.
Use stochastic gradient descent (SGD) to iteratively update embeddings, weights, and biases.
Track loss values to monitor convergence and adjust the learning rate when necessary.

5. Name Generation

Initialize the context with three start tokens.
Perform forward propagation to obtain the next-character probability distribution.
Sample a character using torch.multinomial and update the context window.
Repeat autoregressively until the model generates the end token ..

6. Evaluation & Visualization

Monitor training and validation loss to assess predictive performance and generalization.
Visualize the loss curve and learned character embeddings.
Compare generated names across model architectures to understand the effect of context length and model capacity.

Overall Workflow:
Dataset → Preprocessing → Character Encoding → Context Construction → Embeddings → Neural Network → Softmax → Simultaneous Updates → Generated Names

