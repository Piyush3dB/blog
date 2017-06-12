---
title: Convolutional Neural Networks don't actually perform convolution.
date: "2017-01-20T22:12:03.284Z"
path: "/ConvNet-dont-do-Conv/"
---

Here is a problem: we'd like to convolve convolve image $I$ and kernel $K$ where both signals are 2D defined as

$$
I = 
 \begin{bmatrix}
  1&2&2&1&0 \\\\
  3&3&3&3&0 \\\\
  3&0&0&2&2 \\\\
  0&3&0&3&3 \\\\
  2&2&2&3&2 \\\\
 \end{bmatrix}
$$
$$
K = 
 \begin{bmatrix}
  0 & 1 & 2 \\\\
  3 & 4 & 5 \\\\
  6 & 7 & 8  \\\\
 \end{bmatrix}
$$

A typical computation graph to represent this problem is a single node for the operator and edges representing the tensors

<p align="center">
  <img src="./conv2d.png">
</p>

Which of the following two animations demonstrates $convolve(I, K)$, the one with the <span style="color:blue"> blue </span>kernel or one with the <span style="color:red"> red </span> kernel?

<p align="center">
  <img src="./corr_numerical_no_padding_no_strides.gif">
</p>

<p align="center">
  <img src="./conv_numerical_no_padding_no_strides.gif">
</p>

Both animations show a [MAC operation](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation) being performed as the kernel slides across the image, thew only difference being that the <span style="color:red"> red </span>kernel is a flip re-ordering of the original kernel $K$ colored <span style="color:blue"> blue</span>.

For additional clues to the answer we may turn to the mathematical definition of 2D convolution

$$
C[x,y] = \sum_{j=-1}^{1} \sum_{i=-1}^{1} I[x+i,y+j]K[-i, -j]
$$

The indices are specific to the signal dimensions used here with 0 centering:
* $x,y \in [ -1, 0, 1 ]$ for the result
* $i,j \in [-1, 0, 1 ]$ for the kernel
* $x+i,y+j \in [ -2, -1, 0, 1, 2 ]$ for the input image

The clue is in the orientation of the kernel in the above formula: $\color{red}{K[-i, -j]}$ is a flip re-ordering of $\color{blue}{K[i,j]}$ so the correct answer is the <span style="color:red"> RED</span> animation.


### What do the Deep Learning frameworks implement?

I decided to run a test in [MXNet](http://mxnet.io) to find out what sort of answer I'd get.  The python script below simulates the operator with signals $I$ and $K$ as defined above, expecting a result consistent with the mathematical definition i.e. the result of the <span style="color:red"> red</span> animation.



```python
import mxnet as mx

## Define operator as a computation graph
net  = mx.symbol.Convolution(data=mx.sym.Variable('I'), 
                             num_filter=1, 
                             kernel=(3,3), 
                             name="K")

## Create executor object for operator
c_exec = net.simple_bind(ctx=mx.cpu(), I=(1,1,5,5))

## Copy input arguments to executor space
args={}
args['I'] = mx.nd.array([1,2,2,1,0,
                         3,3,3,3,0,
                         3,0,0,2,2,
                         0,3,0,3,3,
                         2,2,2,3,2]).reshape((1,1,5,5))
args['K_weight'] = mx.nd.array([0,1,2,
                                3,4,5,
                                6,7,8]).reshape((1,1,3,3))
c_exec.copy_params_from(arg_params=args)

## Print arguments
print c_exec.arg_dict['I'].asnumpy()
print c_exec.arg_dict['K_weight'].asnumpy()

## Forward arguments through computation graph
c_exec.forward()

## Show output
print c_exec.outputs[0].asnumpy()

# Result shows a correlation is performed:
#
#[[[[ 60  56  52]
#   [ 39  61  66]
#   [ 54  78  82]]]]
#
```

So according to MXNet, the implementation of convolution does not take into account a flip of the kernel and is infact performing a correlation.  This pattern is not unique to MXNet but is common across all other frameworks.  

So why are the frameworks implementing it in this way when the math clearly states what should be done?  To understand this we have to take a step back and appreciate how convolution and correlation are related.

### More on the Math.

Lets introduce some notation:
* $\color{red}{*}$ to denote <span style="color:red"> convolution</span>.  This is implemented using correlation with a flipped kernel.
* $\color{blue}{\star}$ to denote <span style="color:blue"> correlation</span>.  This is implemented as a sliding dot product.

What is requested when the convolution operator is invoked is identity $\eqref{eq:corrflp}$ but what is actually performed is the RHS of identity $\eqref{eq:corr}$.

$$
  \begin{align}
    \color{red}{convolve}(I, K) &=  \color{blue}{correlate}(I, K_{flipped})
    \label{eq:corrflp}
  \end{align}
$$

$$
  \begin{align}
    \color{red}{convolve}(I, K_{flipped}) &=  \color{blue}{correlate}(I, K)
    \label{eq:corr}
  \end{align}
$$

Using matrices adds more clarity

$$
K_{\color{blue}{corr}} = 
 \begin{bmatrix}
  \color{blue}0 & \color{blue}1 & \color{blue}2 \\\\
  \color{blue}3 & \color{blue}4 & \color{blue}5 \\\\
  \color{blue}6 & \color{blue}7 & \color{blue}8  \\\\
 \end{bmatrix}
$$

is the correlation kernel, then the convolution kernel is derived as

$$
\begin{align\*}
K_{\color{red}{conv}} &= flip\big( K_{\color{blue}{corr}} \big) \\\\
                      &= 
 \begin{bmatrix}
  \color{red}8 & \color{red}7 & \color{red}6 \\\\
  \color{red}5 & \color{red}4 & \color{red}3 \\\\
  \color{red}2 & \color{red}1 & \color{red}0  \\\\
 \end{bmatrix}
\end{align\*}
$$

Identity $\eqref{eq:corr}$ is then re-written as 

$$
I \color{red}{*} K_{\color{red}{conv}} =  I \color{blue}{\star} K_{\color{blue}{corr}}
$$

when expanded 

$$
 \begin{bmatrix}
  1&2&2&1&0 \\\\
  3&3&3&3&0 \\\\
  3&0&0&2&2 \\\\
  0&3&0&3&3 \\\\
  2&2&2&3&2 \\\\
 \end{bmatrix}\color{red}{*}
 \begin{bmatrix}
  \color{red}8 & \color{red}7 & \color{red}6 \\\\
  \color{red}5 & \color{red}4 & \color{red}3 \\\\
  \color{red}2 & \color{red}1 & \color{red}0  \\\\
 \end{bmatrix}=
 \begin{bmatrix}
  1&2&2&1&0 \\\\
  3&3&3&3&0 \\\\
  3&0&0&2&2 \\\\
  0&3&0&3&3 \\\\
  2&2&2&3&2 \\\\
 \end{bmatrix}\color{blue}{\star}
 \begin{bmatrix}
  \color{blue}0 & \color{blue}1 & \color{blue}2 \\\\
  \color{blue}3 & \color{blue}4 & \color{blue}5 \\\\
  \color{blue}6 & \color{blue}7 & \color{blue}8  \\\\
 \end{bmatrix}
$$








scaas







### <span style="color:blue"> Correlation </span>Neural Networks, rather.

It turns out that Convolutional Neural Network is a bit of a misnomer, depending on how you think about it.  Two lines of thought:

1. The 

> The convolution operator in ConvNets is not implemented as convolution, but rather, as correlation.  A more appropriate name for ConvNets is infact CorrNet.




> From a Signal Processing view, convolution is the link that bridges time domain correlation with frequency domain concepts.  Formally this link is known as the Convolution Theorem.





... performing a^[In Digital Signal Processing, the multiply–accumulate operation is fundamental and is the basis upon which dot product procedures are implemented.], also known as the dot product.  



And matrices too:



Some Markdown text with some <span style="color:blue"> *blue* </span>text.

WHat do the deep learning frameworks do.  For this test I've used the [MXNet](http://mxnet.io/) which provides the `mx.symbol.Convolution` operator for this functionality.



Some text in which I cite an author.[^fn1]

More text. Another citation.[^fn2]

What is this? Yet *another* citation?[^fn3]

[^fn1]: So Chris Krycho, "Not Exactly a Millennium," chriskrycho.com, July 22,
    2015, http://www.chriskrycho.com/2015/not-exactly-a-millennium.html
    (accessed July 25, 2015), ¶6.

[^fn2]: Contra Krycho, ¶15, who has everything *quite* wrong.

[^fn3]: ibid.



END