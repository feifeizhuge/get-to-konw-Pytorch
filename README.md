# Get-to-konw-Pytorch
在毕设中需要搭建一个神经网络作为分类器，选用的是Pytorch的框架，在这里记录下我是如何一步一步学习Torch和相关的深度学习的知识。
[写作语法快速参考](https://github.com/guodongxiaren/README#列表)

## 1. What is pytorch 
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

Note: 文档中还介绍了一个常用的函数[torch.view](https://pytorch.org/docs/stable/tensors.html?highlight=view#torch.Tensor.view),它的作用就是reshape Tensor 的维度, 比如`z = x.view(-1, 8)`,就是想把x转换成8列，但是具体多少行，由计算机自己确定，但同时要人为的的保证维度相同。以下是找到的一些参考资料关于这个函数：
* [pytorch中与维度/变换相关的几个函数](https://blog.csdn.net/u013700358/article/details/86301106)
- [Torch张量的view方法有什么作用？](https://vimsky.com/article/3888.html)
* [stackover 上的解释](https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch/42482819#42482819)

#### 2.1.2 What is AUTOGRAD
[第二个文档](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)介绍了自动梯度，这应该是PyTorch这个框架的一个闪亮的特色，目前还是不太懂。
文中介绍了一个简单的例子，就是`z=g(y), y=f(x)`, 通过`z.backward()`就相当于`out.backward(torch.tensor(1.))`, 可以自动的求之前算有运算的梯度，`d(out)/dx`可以通过`x.grad`求得。

[一篇介绍Autograd的博客文章](https://blog.csdn.net/g11d111/article/details/83035270)

[为什么需要zero_grad](https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)


#### 2.1.3 Neural Network for LeNet
[第三个文档](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)介绍了基本的神将网络的结构和运算框图，更加详细的知识还是参考吴恩达老师的课程。
一个神经网络的基本框架:

    input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d

          -> view -> linear -> relu -> linear -> relu -> linear

          -> MSELoss

          -> loss
  
文章中介绍了函数`conv2d`&`maxpool2d`,以前在吴老师课中学习到的函数的底层写法，现在在这里这里调用，理解更加深刻。
* [conv2d](https://pytorch.org/docs/stable/nn.html?highlight=nn.conv2d#torch.nn.Conv2d)
- [maxpool2d](https://pytorch.org/docs/stable/nn.html?highlight=maxpool#torch.nn.MaxPool2d)

如何更新权重，也有函数模块`optimizer`,同时也有`loss function`,等

其中有个特别的参数`dilation`,它叫做空洞卷积：
* [这里](https://blog.csdn.net/hiudawn/article/details/84500648)有它的资料
- 一篇不错的[译文](https://blog.csdn.net/g11d111/article/details/82350563)
* [github上关于不同卷积知识的总结和动图演示](https://github.com/vdumoulin/conv_arithmetic)

关于另一个参数padding，有博文指出并不是直接的补零而是bias，可以参考[这里](https://blog.csdn.net/g11d111/article/details/82665265)

还有一个python的语法`super(Net, self).__init__()`, [什么是super](http://www.runoob.com/python/python-func-super.html)

#### 2.1.4 Training a classifier
[第四篇文档](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)向我们展示了如何训练一个图片分类器，训练的数据都是`Torchvision`已经拆分好了，如果以后用自己的数据，应该还要学习后续的`dataLoader`
