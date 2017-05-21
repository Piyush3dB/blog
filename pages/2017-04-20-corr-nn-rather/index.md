---
title: Convolutional Neural Networks don't perform convolution.
date: "2017-01-20T22:12:03.284Z"
path: "/ConvNet-dont-do-Conv/"
---

Here is a problem: say we wish to convolve a 2D signal $I$ with kernel $K$ each defined as

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

Which of the following two animations demonstrates the convolution procedure?  One with the <span style="color:blue"> blue </span>kernel or one with the <span style="color:red"> red </span> kernel?

<p align="center">
  <img src="./corr_numerical_no_padding_no_strides.gif">
</p>

<p align="center">
  <img src="./conv_numerical_no_padding_no_strides.gif">
</p>

The animations show a kernel sliding across the signal, at each spatial location performing a [MAC operation](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation)^[In Digital Signal Processing, the multiply–accumulate operation is fundamental and is the basis upon which dot product procedures are implemented.], also known as the dot product.  Observe that the <span style="color:red"> red </span>kernel is a flip re-ordering of the <span style="color:blue"> blue </span>kernel.

The correct answer is infact the one with the <span style="color:red"> red </span>kernel, and this can be verified by the mathematical definition of convolution in this case:


$$
P_{ij} = \sum_{23}^{123}
$$






finite support in the set $\\{ -3, -2, -1, 0, 1, 2, 3 \\}$ and the kernel has support in $\\{-1, 0, 1 \\}$

$\text{Look at the }\mathtt{\{}\text{braces}\mathtt{\}}\texttt{.}$

The kernel is positioned along the signal


The key 


## Mathematical Equations

Inline math with special characters: $|\psi\rangle$, $\Omega'$, $\gamma^\*$.  Bayes formula is $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$.


In equation $\eqref{eq:sample}$, we find the value of an
interesting integral:

$$
\star *
\begin{equation}
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
  \label{eq:sample}
\end{equation}
$$


Bigger equations:

$$
\begin{align}
E(\mathbf{v}, \mathbf{h}) = -\sum_{i,j}w_{ij}v_i h_j - \sum_i b_i v_i - \sum_j c_j h_j
\end{align}
$$

In multiline is:

$$
\begin{align}
                p(v_i=1|\mathbf{h}) & = \sigma\left(\sum_j w_{ij}h_j + b_i\right) \\\\
                p(h_j=1|\mathbf{v}) & = \sigma\left(\sum_i w_{ij}v_i + c_j\right)
\end{align}
$$


And without numbering:
$$
  \begin{align\*}
    |\psi_1\rangle &= a|0\rangle + b|1\rangle \\\\
    |\psi_2\rangle &= c|0\rangle + d|1\rangle
  \end{align\*}
$$

Miltiline alignment:
$$
\begin{equation} 
\begin{split}
A & = \frac{\pi r^2}{2} \\\\
  & = \frac{1}{2} \pi r^2 \\\\
  & = \frac{A}{B_{C}} \psi r^{\theta} \\\\
\end{split}
\end{equation}
$$

And matrices too:
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
I \color{red}{*} K_{\color{red}{conv}} =  I \color{blue}{\star} K_{\color{blue}{corr}}
$$


Some Markdown text with some <span style="color:blue"> *blue* </span>text.

WHat do the deep learning frameworks do.  For this test I've used the [MXNet](http://mxnet.io/) which provides the `mx.symbol.Convolution` operator for this functionality.



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

# Result ------>
#
#[[[[ 60.  56.  52.]
#   [ 39.  61.  66.]
#   [ 54.  78.  82.]]]]
#

```


Some text in which I cite an author.[^fn1]

More text. Another citation.[^fn2]

What is this? Yet *another* citation?[^fn3]

[^fn1]: So Chris Krycho, "Not Exactly a Millennium," chriskrycho.com, July 22,
    2015, http://www.chriskrycho.com/2015/not-exactly-a-millennium.html
    (accessed July 25, 2015), ¶6.

[^fn2]: Contra Krycho, ¶15, who has everything *quite* wrong.

[^fn3]: ibid.



END