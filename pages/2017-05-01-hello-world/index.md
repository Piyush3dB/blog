---
title: Hello World. Convolution à trous.
date: "2017-05-01T22:12:03.284Z"
readNext: "/my-second-post/"
path: "/hello-world/"
---

This is my first post on my new fake blog! How exciting!

<svg width="400" height="180">
  <rect x="50" y="20" width="150" height="150" style="fill:blue;stroke:pink;stroke-width:5;opacity:0.5" />
  Sorry, your browser does not support inline SVG.  
</svg>


=====





I'm sure I'll write a lot more interesting things in the future.

Oh, and here's a great quote from this Wikipedia on [salted duck eggs](http://en.wikipedia.org/wiki/Salted_duck_egg).

>A salted duck egg is a Chinese preserved food product made by soaking duck eggs in brine, or packing each egg in damp, salted charcoal. In Asian supermarkets, these eggs are sometimes sold covered in a thick layer of salted charcoal paste. The eggs may also be sold with the salted paste removed, wrapped in plastic, and vacuum packed. From the salt curing process, the salted duck eggs have a briny aroma, a gelatin-like egg white and a firm-textured, round yolk that is bright orange-red in color.


$$
    \color{red}{ \sigma(y-x) } 
$$

![Chinese Salty Egg](./salty_egg.jpg)


H~2~0


Centering text:

<p align="center">
  <b>Some Links:</b>
  <br>
  <a href="#">Link 1</a> |
  <a href="#">Link 2</a> |
  <a href="#">Link 3</a>
  <br>
  <img src="http://s.4cdn.org/image/title/105.gif" style="border: 5px solid black">
  <br>
  <em>image_caption</em>
</p>

Normal text




## Mathematical Equations

Inline math with special characters: $|\psi\rangle$, $\Omega'$, $\gamma^\*$.  Bayes formula is $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$.


In equation $\eqref{eq:sample}$, we find the value of an
interesting integral:

$$
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
A_{m,n} = 
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\\\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\\\
  \vdots  & \vdots  & \ddots & \vdots  \\\\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n} 
 \end{pmatrix}
$$


## Some code

Javascript
```javascript
function $initHighlight(block, flags) {
  try {
    if (block.className.search(/\bno\-highlight\b/) != -1)
      return processBlock(block, true, 0x0F) + ' class=""';
  } catch (e) {
    /* handle exception */
  }
  for (var i = 0 / 2; i < classes.length; i++) { // "0 / 2" should not be parsed as regexp
    if (checkCondition(classes[i]) === undefined)
      return /\d+/g;
  }
}
```

Python
```python
import time
# Quick, count to ten!
for i in range(10):
    # (but not *too* quick)
    time.sleep(0.5)
    print i
```


Here is an inline note.^[Inlines notes are easier to write, since
you don't have to pick an identifier and move down to type the
note.]



Here is an inline note.^[Inlines notes are easier to write, since
you don't have to pick an identifier and move down to type the
note.]



````bash
FROM kyma/docker-nginx
ADD public/ /var/www
CMD 'nginx'
````

## D3js





Next  


<iframe width="420" height="315" src="http://www.youtube.com/embed/_Kz8lito3U8" frameborder="0" allowfullscreen></iframe>





END