---
title: Writing Gatsby Articles in Markdown and LaTeX
date: 2020-03-05 00:00:00 +0300
description: Writing a math article in Gatsby, Latex and Markdown.
img: ./maths.jpg # Add image post (optional)
tags: [Syntax, Markdown, Latex] # add tag
---

Markdown is designed to be an easy-to-read and easy-to-write markup language. This is not always the case, as it can be a daunting challenge to write and style a markdown article that has a mix of **both code and maths**. In this article, I have put together the commonly used syntax for scientific writing in markdown.

**Install**

npm install --save gatsby-transformer-remark gatsby-remark-katex katex

**gatsby-config.js**

plugins:[
	{
		resolve: `gatsby-transformer-remark`, 
		options: {
		plugins: [
			{
				resolve: `gatsby-remark-katex`, 
				options: {
				strict: `ignore`
			}
			}
		]
	}
	}
]

**layout.js**

require(`katex/dist/katex.min-css`)

## Jump to: 
* [Python code in Markdown](#code)
* [Pure Markdown syntax](#mark)
* [Latex in Markdown](#latex)
* [Further reading](#further)

## Python, javascript, html and bash <a name="code"></a>code in Markdown 

### Python
```python

	def add(x,y):
		return x+y
```
'''python
	def add(x, y):
		return x+y
'''

***Replace the python keyword with js, html, or bash for javascript, html or bash code respectively.***


### Javascript

```javascript{numberLines: true}

console.log("Hello World!!!");

```

### Html

```html

<b> Bold text </b>

```

### Bash
```bash{promptUser: steve} 

55

```
## Writing in Pure <a name="mark"></a>Markdown

### Headers
# Header 1
##### Header 5
```javascript
# Header 1 
###### Header 5
```
\# escaping header
```js
\# escaping header

```
### Bold and Italic
**bold**
```js
**bold**
__bold__

```
*italic*
```js
*italic*
_italic_

```
***bold and italic***
```js
***bold and italic***

```
~~Strike thru~~
```js
~~Strike thru~~

```
### Paragraph, blockquote and horizontal line
paragraph 1

paragragh 2
```js
paragrah 1

blank line also means break to new line 

```
> This is a blockquote
>> This is a deeper blockquote
>>> This is a deepest blockquote
```js
> blockquote
>> deeper blockquote
>>> deepest blockquote

```

---
___

***
```js
---
___

***

```

### Ordered and unordered Lists

1. Dog
2. Cat

```js
1. Dog
2. Cat

```
- Dog
- Cat

```js
- Dog
- Cat

```

### Images

![](./pic.jpg "Simba")


```js
![](./pic.jpg "Simba")
```
### Anchor, navigation and reference links

https://www.google.com

```js

https://www.google.com
```
[Google](https://www.google.com)
```js
[Google](https://www.google.com)
```

Click a referenced [ref1] url

[ref1]: https://www.google.com "Google"

```js
Click a referenced [ref1] url

[ref1]: https://www.google.com "Google"

```


[Title Link has no spaces (hyphen)](#id-1)

```js
[Title Link has no spaces (hyphen)](#id-1)

```
<nav> 
	<a href="https://www.google.com"> google </a> |
	<a href="https://www.amazon.com"> amazon </a>
</nav>

```html
<nav> 
	<a href="https://www.google.com"> google </a> |
	<a href="https://www.amazon.com"> amazon </a>
</nav>
```
#### Table of Contents
* [Chapter 1](#chap-1)
* [Chapter 2](#chap-2)

##### Chapter 1 <a name="chap-1"></a> Content one 
##### Chapter 2 <a name="chap-2"></a> Content two 

```js
#### Table of Contents
* [Chapter 1](#chap-1)
* [Chapter 2](#chap-2)

##### Chapter 1 <a name="chap-1"></a> Content one 
##### Chapter 2 <a name="chap-2"></a> Content two
```


### Todo
- [x] Task 1
- [ ] Task 2

```js
- [x] Task 1
- [] Task 2

```

### Table
|Key |Value| Type |
|----|:---:|-----:|
|name   |Simba| str |
|age  | 18 | int  |

```js
|Key |Value| Type |
|----|:---:|-----:|
|name   |Simba| str |
|age  | 18 | int  |
```
### Color and font
<span style="color: green">green text</span>
```js
<span style="color: green">green text</span>
```

<span style="color: blue; font-family: 'Bebas Neue'; font-size: 2em;">text : 23</span>

```js
<span style="color: blue; font-family: 'Bebas Neue'; font-size: 2em;">
text : 23</span>
```

<code style="background:yellow; color: red">Highlight some text </code>

```js
<code style="background:yellow; color: red">
Highlight some text </code>

```


## LaTeX : Math<a name="latex"></a> Notation 

### inline:    $a^2 + b^2 = c^2$

```javascript
$ a^2 + b^2 = c^2 $
```

### Multiline
$$
a^2 + b^2 = c^2 
$$

```javascript
$$
a^2 + b^2 = c^2 
$$

```


### Calculus
$$
f(x) = \int_{-infty}^\infty
\hat f(\xi)\,e^{2 \pi i \xi x}
\,d\xi
$$
```javascript
$$	
f(x) = \int_{-\infty}^\infty
\hat f(\xi)\,e^{2 \pi i \xi x}
 \,d\xi
$$
```
$$
\int u \frac{dv}{dx}\,dx=uv-\int \frac{du}{dx}v\,dx
$$
```js
\int u \frac{dv}{dx}\,dx=uv-\int \frac{du}{dx}v\,dx
```
$$
\oint \vec{F} \cdot d\vec{s}=0
$$
```js
\oint \vec{F} \cdot d\vec{s}=0
```
### Fractions on fractions
$$
\frac{\frac{1}{x}+\frac{1}{y}}{y-z}
$$
```js
\frac{\frac{1}{x}+\frac{1}{y}}{y-z}

```

### Repeating Fractions
$$
	\frac{1}{\Bigl(\sqrt{\phi \sqrt{5}}-\phi\Bigr) e^{\frac25 \pi}} = 1+\frac{e^{-2\pi}} {1+\frac{e^{-4\pi}} {1+\frac{e^{-6\pi}} {1+\frac{e^{-8\pi}} {1+\cdots} } } }
$$

```javascript
\frac{1}{\Bigl(\sqrt{\phi \sqrt{5}}-\phi\Bigr) e^{\frac25 \pi}} = 
1+\frac{e^{-2\pi}} {1+\frac{e^{-4\pi}} {1+\frac{e^{-6\pi}} 
{1+\frac{e^{-8\pi}} {1+\cdots} } } }

```
### Greek Letters:  
$$
	\Gamma\ \Delta\ \Theta\ \Lambda\ \Xi\ \Pi\ \Sigma\ \Upsilon\ \Phi\ \Psi\ \Omega
$$
$$
	\alpha\ \beta\ \gamma\ \delta\ \epsilon\ \zeta\ \eta\ \theta\ \iota\ \kappa\ \lambda\ \mu\ \nu\ \xi\ \omicron\ \pi\ \rho\ \sigma\ \tau\ \upsilon\ \phi\ \chi\ \psi\ \omega\ \varepsilon\ \vartheta\ \varpi\ \varrho\ \varsigma\ \varphi
$$
```js
\Gamma\ \Delta\ \Theta\ \Lambda\ \Xi\ \Pi\ \Sigma\
 \Upsilon\ \Phi\ \Psi\ \Omega
\alpha\ \beta\ \gamma\ \delta\ \epsilon\ \zeta\ \eta\ 
\theta\ \iota\ \kappa\ \lambda\ \mu\ \nu\ \xi\ \omicron\
\pi\ \rho\ \sigma\ \tau\ \upsilon\ \phi\ \chi\ \psi\ \omega\
 \varepsilon\ \vartheta\ \varpi\ \varrho\ \varsigma\ \varphi
```

### Symbols
$$
\surd\ \barwedge\ \veebar\ \odot\ \oplus\ \otimes
 \oslash\ \circledcirc\ \boxdot\ \bigtriangleup
$$
```js
\surd\ \barwedge\ \veebar\ \odot\ \oplus\ \otimes
 \oslash\ \circledcirc\ \boxdot\ \bigtriangleup
```

### Lorenz Equations: 
$$
	\begin{aligned}
		\dot{x} & = \sigma(y-x) \\
		\dot{y} & = \rho x - y - xz \\
		\dot{z} & = -\beta z + xy
	\end{aligned}
$$

```javascript
\begin{aligned}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{aligned}
```

### Maxwell's Equations
$$
\begin{aligned}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\[1em]
\nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\[0.5em]
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\[1em]
\nabla \cdot \vec{\mathbf{B}} & = 0 \end{aligned}
$$

```js
\begin{aligned}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, 
\frac{\partial\vec{\mathbf{E}}}{\partial t} & = 
\frac{4\pi}{c}\vec{\mathbf{j}} \\[1em]
\nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\[0.5em]
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, 
\frac{\partial\vec{\mathbf{B}}}{\partial t} & = 
\vec{\mathbf{0}} \\[1em]
\nabla \cdot \vec{\mathbf{B}} & = 0 \end{aligned}
```

### Matrices:
$$
\begin{pmatrix}
a_{11} & a_{12} & a_{13}\\ 
a_{21} & a_{22} & a_{23}\\ 
a_{31} & a_{32} & a_{33}
\end{pmatrix}
$$
```js
\begin{pmatrix}
a_{11} & a_{12} & a_{13}\\ 
a_{21} & a_{22} & a_{23}\\ 
a_{31} & a_{32} & a_{33}
\end{pmatrix}
```

$$
 \begin{bmatrix} A_{1,1} & A_{1,2} \\ \colorbox{blue} A_{2,1} & \color{blue}A_{2,2} \\ A_{3,1} & A_{3,2} \end{bmatrix}+ 
 \begin{bmatrix} B_{1,1} & B_{1,2} \\ B_{2,1} & B_{2,2} \\ B_{3,1} & B_{3,2} \end{bmatrix}= 
 \begin{bmatrix} A_{1,1} + B_{1,1} & A_{1,2} + B_{1,2} \\ A_{2,1} + B_{2,1} & A_{2,2} + B_{2,2} \\ A_{3,1} + B_{3,1} & A_{3,2} + B_{3,2} \end{bmatrix} 
\begin{Bmatrix} \end{Bmatrix} 
\begin{pmatrix} \end{pmatrix}

$$

```js
 \begin{bmatrix} A_{1,1} & A_{1,2} \\ \colorbox{blue} A_{2,1} &  \color{blue} A_{2,2} \\ A_{3,1} & A_{3,2} \end{bmatrix}+ 
 \begin{bmatrix} B_{1,1} & B_{1,2} \\ B_{2,1} & B_{2,2} \\ B_{3,1} & B_{3,2} \end{bmatrix}= 
 \begin{bmatrix} A_{1,1} + B_{1,1} & A_{1,2} + B_{1,2} \\ A_{2,1} + B_{2,1} & A_{2,2} + B_{2,2} \\ A_{3,1} + B_{3,1} & A_{3,2} + B_{3,2} \end{bmatrix} 
\begin{Bmatrix} \end{Bmatrix} 
\begin{pmatrix} \end{pmatrix}

```
$$
\begin{bmatrix} 0 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 0 \end{bmatrix}
$$
```js
\begin{bmatrix} 0 & \cdots & 0 \\ \vdots & 
\ddots & \vdots \\ 0 & \cdots & 0 \end{bmatrix}
```

### Inline Statements

$\huge You$ can $\tiny also $ make use of $\Large inline $ statements. The formula $$ e^{\pi i} + 1 = 0 $$ is nicely produced within this text, as well as $$ 5 + 5 \: = \; 10 $$, $$$A_{ij} + B{i,j} = C_{i,j}$$ and  $$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$.
Also $$x_1$$ and $$\left.\frac{x^3}{3}\right|_0^1$$ works well. Another example is $$\lim_{x \to \infty} \exp(-x) = 0$$ or $$\frac{\frac{1}{x}+\frac{1}{y}}{y-z}$$ or $$\cos (2\theta) = \cos^2 \theta - \sin^2 \theta$$ or $$\left.\frac{x^3}{3}\right\vert_0^1$$ or $$P\left(A=2\middle\vert\frac{A^2}{B}>4\right)$$. The inner product given $x, y \in \mathbb R^n$ is $x'y = \sum_{i=1}^n x_i y_i$.


```js
$\huge You $ can $\tiny also $ make use of $\Large inline $ statements. The formula $$ e^{\pi i} + 1 = 0 $$ is nicely produced within this text, as well as $$ 5 + 5 \: =  \; 10 $$, $$ $A_{ij} + B{i,j} = C_{i,j} $$ and  $$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$. Also $$x_1$$ and $$\left.\frac{x^3}{3}\right|_0^1$$ works well. Another example is $$ \lim_{x \to \infty} \exp(-x) = 0$$ or $$\frac{\frac{1}{x}+\frac{1}{y}}{y-z}$$ or $$\cos (2\theta) = \cos^2 \theta - \sin^2 \theta$$ or $$\left.\frac{x^3}{3}\right\vert_0^1$$ or $$P\left(A=2\middle\vert\frac{A^2}{B}>4\right)$$.The inner product given $x, y \in \mathbb R^n$ is $xy = \sum_{i = 1}^n x_i y_i$.
```
### Sum of Series

$$
\displaystyle\sum_{i=1}^{k+1}i
\displaystyle= \left(\sum_{i=1}^{k}i\right) +(k+1)
$$
```js
$$
\displaystyle\sum_{i=1}^{k+1}i
\displaystyle= \left(\sum_{i=1}^{k}i\right) +(k+1)
$$
```

### Product Notation
$$
\displaystyle1 +  \frac{q^2}{(1-q)}+\frac{q^6}{(1-q)(1-q^2)}+\cdots
= \displaystyle \prod_{j=0}^{\infty}\frac{1}{(1-q^{5j+2})(1-q^{5j+3})},
\displaystyle\text{ for }\lvert q\rvert < 1.
$$
```js
$$
\displaystyle1 +  \frac{q^2}{(1-q)}+\frac{q^6}{(1-q)(1-q^2)}+\cdots
= \displaystyle \prod_{j=0}^{\infty}\frac{1}{(1-q^{5j+2})(1-q^{5j+3})},
\displaystyle\text{ for }\lvert q\rvert < 1.
$$

```
### Cross Product
$$
\mathbf{V}_1 \times \mathbf{V}_2  \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0
\end{vmatrix}
$$
```html
\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0
\end{vmatrix}
```


### Arrows
$$
\gets\ \to\ \leftarrow\ \rightarrow\ \uparrow\ \Uparrow\ \downarrow\ \Downarrow\ \updownarrow\ \Updownarrow
$$
```js
$$
\gets\ \to\ \leftarrow\ \rightarrow\ \uparrow\ \Uparrow\ 
\downarrow\ \Downarrow\ \updownarrow\ \Updownarrow
$$
```
$$
\Leftarrow\ \Rightarrow\ \leftrightarrow\ \Leftrightarrow 
\mapsto\ \hookleftarrow
$$
```js
\Leftarrow\ \Rightarrow\ \leftrightarrow\ \Leftrightarrow 
\mapsto\ \hookleftarrow
```
$$
\leftharpoonup\ \leftharpoondown\ \rightleftharpoons\ \longleftarrow\ \Longleftarrow\ \longrightarrow
$$
```js
\leftharpoonup\ \leftharpoondown\ \rightleftharpoons
 \longleftarrow\ \Longleftarrow\ \longrightarrow
```

### Accents
$$
\hat{x}\ \vec{x}\ \ddot{x}
$$
```js
\hat{x}\ \vec{x}\ \ddot{x}
```

### Evaluation at limits
$$
\left.\frac{x^3}{3}\right|_0^1
$$
```js
\left.\frac{x^3}{3}\right|_0^1
```
### Case Definitions
$$
f(n) = \begin{cases} \frac{n}{2}, & \text{if } n\text{ is even} \\ 3n+1, & \text{if } n\text{ is odd} \end{cases}
$$
```js
f(n) = \begin{cases} \frac{n}{2}, & \text{if } n\text{ is even} 
\\ 3n+1, & \text{if } n\text{ is odd} \end{cases}
```
### Statistics
$$
f(n) = \begin{cases} \frac{n}{2}, & \text{if } n\text{ is even} \\ 3n+1, & \text{if } n\text{ is odd} \end{cases}
$$
```js
f(n) = \begin{cases} \frac{n}{2}, & \text{if } n\text{ is even} 
\\ 3n+1, & \text{if } n\text{ is odd} \end{cases}
```
$$
{n \choose k}
$$
```js
{n \choose k}
```
### Punctuation
$$
f(x) = \sqrt{1+x} \quad (x \ge  -1)
$$
```js
f(x) = \sqrt{1+x} \quad (x \ge  -1)
```
$$
f(x) \sim x^2 \quad (x\to\infty)
$$
```js
f(x) \sim x^2 \quad (x\to\infty)
```
$$
f(x) = \sqrt{1+x}, \quad x \ge -1
$$
```js
f(x) = \sqrt{1+x}, \quad x \ge -1
```
$$
f(x) \sim x^2, \quad x\to\infty
$$
```js
f(x) \sim x^2, \quad x\to\infty
```
$$
\mathcal L_{\mathcal T}(\vec{\lambda})
    = \sum_{(\mathbf{x},\mathbf{s})\in \mathcal T}
       \log P(\mathbf{s}\mid\mathbf{x}) - \sum_{i=1}^m
       \frac{\lambda_i^2}{2\sigma^2}
$$
```js
\mathcal L_{\mathcal T}(\vec{\lambda})
    = \sum_{(\mathbf{x},\mathbf{s})\in \mathcal T}
       \log P(\mathbf{s}\mid\mathbf{x}) - \sum_{i=1}^m
       \frac{\lambda_i^2}{2\sigma^2}
```
$$
S (\omega)=\frac{\alpha g^2}{\omega^5} \,
e ^{[-0.74\bigl\{\frac{\omega U_\omega 19.5}
{g}\bigr\}^{-4}]}
$$

```js
S (\omega)=\frac{\alpha g^2}{\omega^5} \,
e ^{[-0.74\bigl\{\frac{\omega U_\omega 19.5}
{g}\bigr\}^{-4}]}
```

### Notatition

$
 a \: \in \: A 
 $
  : ants in Animal Kingdom // a elements belonging to A
 
 $
 B \subset A 
 $
  : every element of B found in A (subset) // inclusion
 
 $
 : or |  \;
 $
 : such that e.g. $ B = \begin{Bmatrix} a \: \in  \: A : a \: is \: ants \end{Bmatrix}$

$
\begin{pmatrix} x \: R \: y \end{pmatrix}
$
 : relation between x and y

$
 dom \: R = \begin{Bmatrix} x : \: for \: some \: y \: \begin{pmatrix} x \: R \: y \end{pmatrix} \end{Bmatrix}
$
 : a domain relation formed by a set of input values of x, at least one value of x has a relation to y // domain (inputs) and range (outputs).

$
f :  \: X \rightarrow Y  \; 
\; or \; \; f(x) \: = \: y \; \; or \; \;  \begin{pmatrix} x, y \end{pmatrix} \: \in \: f 
$
: a function maps or transforms

$
:= 
$
 means defined as

#### Logical Symbols

 $
 \wedge 
 $
 : and (conjuction)

 $
 \vee
 $
 : or (disjucntion)

 $
 \exists
 $
 : some

 $
 \forall
 $
 : all

 $
 \neg
 $
 : negates

 $
 \rightarrow
 $
 : implication ie. $ p \rightarrow q$ ***means*** if p is true, then q is also true

 $
 \leftrightarrow
 $
 : bicondition ie. $ p \leftrightarrow q$ ***means*** both are jointly true or both are jointly false



***

## Further <a name="further"></a>Reading
- [Markdown Cheatsheet](https://assemble.io/docs/Cheatsheet-Markdown.html)
- [The Ultimate Markdown Guide (for Jupyter Notebook)](https://medium.com/analytics-vidhya/the-ultimate-markdown-guide-for-jupyter-notebook-d5e5abf728fd)
- [Latex Community](https://latex.org/forum/app.php/page/blogs)
- [KaTeX and MathJax Comparison Demo](https://www.intmath.com/cg5/katex-mathjax-comparison.php)
- [Introduction to Linear Algebra for Applied Machine Learning](https://pabloinsente.github.io/)







