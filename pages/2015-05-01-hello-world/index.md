---
title: Hello World
date: "2015-05-01T22:12:03.284Z"
readNext: "/my-second-post/"
path: "/hello-world/"
---

This is my first post on my new fake blog! How exciting!

I'm sure I'll write a lot more interesting things in the future.

Oh, and here's a great quote from this Wikipedia on [salted duck eggs](http://en.wikipedia.org/wiki/Salted_duck_egg).

>A salted duck egg is a Chinese preserved food product made by soaking duck eggs in brine, or packing each egg in damp, salted charcoal. In Asian supermarkets, these eggs are sometimes sold covered in a thick layer of salted charcoal paste. The eggs may also be sold with the salted paste removed, wrapped in plastic, and vacuum packed. From the salt curing process, the salted duck eggs have a briny aroma, a gelatin-like egg white and a firm-textured, round yolk that is bright orange-red in color.

![Chinese Salty Egg](./salty_egg.jpg)




## Mathematical Equations

Inline math with special characters: $|\psi\rangle$, $\Omega'$, $\gamma^\*$.  Bayes formula is $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$.

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





## D3js



<svg  xmlns="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink">
    <rect x="10" y="10" height="100" width="100"
          style="stroke:#ff0000; fill: #0000ff"/>
</svg>



<iframe src="https://cdn.rawgit.com/espinielli/75209760a6d8f8922e9c/raw/9ac287b7c808d19eba58ee4a162985e1d8bf5121/index.html" marginwidth="0" marginheight="0" scrolling="no"></iframe>


Next  


<iframe src="http://bl.ocks.org/mbostock/raw/4061502/0a200ddf998aa75dfdb1ff32e16b680a15e5cb01/" marginwidth="0" marginheight="0" scrolling="no"></iframe>


Next  


<iframe width="420" height="315" src="http://www.youtube.com/embed/_Kz8lito3U8" frameborder="0" allowfullscreen></iframe>




ROC:

<div class="layout-wrapper">
    <div class="controls">
        <label for="mean1">mean #1:</label><input id="mean1" type = "number" size = "5" value = "0" onchange="draw()">
        <label for="mean2">mean #2:</label><input id="mean2" type = "number" size = "5" value = "2" onchange="draw()">
        <label for="var1">variance #1:</label><input id="var1" type = "number" size = "5" value = "4" onchange="draw()">
        <label for="var2">variance #2:</label><input id="var2" type = "number" size = "5" value = "4" onchange="draw()">
    </div>
    <div id="renderer">
        <!-- here all the plots will be rendered -->
    </div>

    <link rel="stylesheet" href="/css/roc_curve.css">
    <script src="/scripts/d3.min.js" charset="utf-8"></script>
    <script src="/scripts/jquery-2.1.4.js" charset="utf-8"></script>
    <script src="/scripts/roc_curve.js" charset="utf-8"></script>
</div>

END