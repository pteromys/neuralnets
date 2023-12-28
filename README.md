# Neural network toy

Tamper with hyperparameters as a simple neural network trains.

![screenshot of the neural network toy](preview2d.jpg "Screenshot")

Just go to the [live demo](https://pteromys.melonisland.net/neuralnets/).

## Implementation

* backpropagation from scratch in nn.js, self-contained but unsophisticated
* fragile bit-packing to run inference as a fragment shader
* jQuery to smooth over the usual papercuts

## License

[MIT](./LICENSE)

## See also / prior art

* [Andrej Karpathy's ConvNetJS demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)
  graphs where the input grid lands in each layer's output space.
* [Tensorflow's playground](https://playground.tensorflow.org/)
  graphs each neuron's output as a function of the 2D input space,
  including parts where the sample data doesn't cover.
