# How to generate cactus distribution?

## cactus generators

There are two functions in this repo, cactus_generator.m and cactus_generator_monotonic.m. To generate the desired cactus distribution which given the best privacy performance under given cost constraints, you need to set configurations such as sensitivity, desired cost (l1 or l2), and quantization parameters, etc.. More details please refer to the comments within the code.

## test in TensorFlow
For running this code, it is required to have TensorFlow 2.4 or later version ready on your machine. The file dp_optimizer_keras_vectorized.py is a modified version of the original file in TensorFlow, which replaces the gaussian noise with the corresponding cactus noise. For speeding the code, we pre-generated a pool of samples stored in TestData folder, and each time the function may uniformly at random pick a sample from the pool.
