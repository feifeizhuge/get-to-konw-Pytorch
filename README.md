# get-to-konw-Pytorch
在毕设中需要搭建一个神经网络作为分类器，选用的是Pytorch的框架，在这里记录下我是如何一步一步学习Torch和相关的深度学习的知识。
[写作语法快速参考](https://github.com/guodongxiaren/README#列表)

## 1. what is pytorch 
Todo

## 2. PyTorch official Tutorial
在官方网站上，已经有现成的入门教程提供给了我们，第一个就是“60分钟入门教程”，接下来我们就来一起学习这个教程，并发现其中的一些隐藏的知识。
### 2.1 Deep learning with PyTorch: a 60 minute blitz
说得是60分钟，但仔细认真的看下来至少得4个小时😂，[这里](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)可以参考官网的教程，同时还有热心的知乎网友翻译的[入门教程](https://zhuanlan.zhihu.com/p/25572330)(不过不是最新的)
#### 2.1.1 What is PyTorch 
[第一个文档](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)介绍了一个新的数据结构 Tensor， 我认为它就是一个多维的数组，至于为什么要有它呢？😑应该是它可以放在GPU里面加速运算。
这个文档的知识结构如下:   
* 数据结构Tensor, 它是什么, 如何初始化&赋值
- basic operation for Tensor, klick [here](https://pytorch.org/docs/stable/torch.html) and you can see more details
* numpy 与 Tensor 之间的相互转换
- Cuda Tensor

Note: 文档中还介绍了一个常用的函数[torch.view](),它的作用就是reshape Tensor 的维度, 比如`z = x.view(-1, 8)`,就是想把x转换成8列，但是具体多少行，由计算机自己确定，但同时要人为的的保证维度相同。以下是找到的一些参考资料关于这个函数：
* [pytorch中与维度/变换相关的几个函数](https://blog.csdn.net/u013700358/article/details/86301106)
- [Torch张量的view方法有什么作用？](https://vimsky.com/article/3888.html)
* [stackover 上的解释](https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch/42482819#42482819)
