---
title: Convolutional Neural Networks don't actually perform convolution.
date: "2017-01-20T22:12:03.284Z"
path: "/ConvNet-dont-do-Conv/"
---

Here is a problem: say we would like to convolve 2D signal $I$ with kernel $K$ defined as

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


<p align="center">
  <img src="./conv2d.png">
</p>

Which of the following two animations demonstrates the convolution procedure? i.e.  

$$
  convolve(I, K)
$$

One with the <span style="color:blue"> blue </span>kernel or one with the <span style="color:red"> red </span> kernel?

<p align="center">
  <img src="./corr_numerical_no_padding_no_strides.gif">
</p>

<p align="center">
  <img src="./conv_numerical_no_padding_no_strides.gif">
</p>


The animations show a [MAC operation](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation) being performed as the kernel slides across the signal, but notice that the <span style="color:red"> red </span>kernel is a flip re-ordering of the <span style="color:blue"> blue </span>kernel. 

Mathematically the 2D convolutuion procedure is written as

$$
C[x,y] = \sum_{j=-1}^{1} \sum_{i=-1}^{1} I[x+i,y+j]K[-i, -j]
$$


where the output $C$ has finite support in the set $x,y \in \\{ -1, 0, 1 \\}$ and the kernel has support $x,y \in \\{-1, 0, 1 \\}$. $x+i,y+j \in \\{ -2, -1, 0, 1, 2 \\}$

where $\color{red}{K[-i, -j]}$ is the flip of $\color{blue}{K[i,j]}$.


So answer to the question is the animation one with the <span style="color:red"> red </span>kernel and this is confirmed by the mathematical definition.


### What do the Deep Learning frameworks implement?


Hold this thought in mind and think about what the Deep Learning frameworks implement for the Convolution operator.  I decided to test this in [MXNet](http://mxnet.io) using the python script below.  We expect this script to produce an aoutput that is consistent with the mathematical definition i.e. result with the <span style="color:red"> red </span>kernel.



```python
## Import mxnet python module
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

# Result shows the blue kernel is used:
#
#[[[[ 60  56  52]
#   [ 39  61  66]
#   [ 54  78  82]]]]
#
```

So according to MXNet, the implementation of convolution does not take into account a flip of the kernel and is infact performing the animation with the <span style="color:blue"> blue </span>kernel above.  This pattern is not unique to MXNet but is common across all other frameworks.




### Correlation Neural Networks, rather.



> The Convolution Operator in ConvNets don't perform Convolution, but rather, perform Correlation because the kernel is not flipped.




> Convolution is the link that bridges Correlation with Frequency Domain Theory.  In Signal Processing theory this link is known as the Convolution Theorem.





... performing a^[In Digital Signal Processing, the multiply–accumulate operation is fundamental and is the basis upon which dot product procedures are implemented.], also known as the dot product.  



Notation:
* to denote $\color{red}{convolution}$ : $\color{red}{*}$
* to denote $\color{blue}{correlation}$ : $\color{blue}{\star}$


And matrices too:




$$
  \begin{align\*}
    \color{red}{convolve}(I, K) &=  \color{blue}{correlate}(I, K_{flipped}) \\\\
    \color{red}{convolve}(I, K_{flipped}) &=  \color{blue}{correlate}(I, K)
  \end{align\*}
$$



$$
K_{\color{blue}{corr}} = 
 \begin{bmatrix}
  \color{blue}0 & \color{blue}1 & \color{blue}2 \\\\
  \color{blue}3 & \color{blue}4 & \color{blue}5 \\\\
  \color{blue}6 & \color{blue}7 & \color{blue}8  \\\\
 \end{bmatrix}
$$

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




$$
I \color{red}{*} K_{\color{red}{conv}} =  I \color{blue}{\star} K_{\color{blue}{corr}}
$$


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