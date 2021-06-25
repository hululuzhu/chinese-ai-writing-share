# chinese-ai-writing-share
中文AI写作共享

## 声明
- 基于主流transformer的业余水平语言模型实现，但因为几小时见效，故适合自娱自乐
- 数据质量直接影响模型效果，如模型有异常输出（如黄赌毒爆等内容），请大家自动忽略，不要报警

## 架构
- 基于Transformer的encoder-decoder
- transformer使用keras-transformer lib

## 写诗
- 数据来源 https://github.com/chinese-poetry/chinese-poetry

## 对联
- 数据来源 https://github.com/wb14123/couplet-dataset

## 训练
- 推荐Google Colab Pro （16G的GPU一个月随便用才9.99！）
- TPU的支持有空会跟上

## 保存
- 训练完的模型可以直接`model.save(path)`，但不知为何有问题，所以我存了H5参数
- 另外用pickle保存vocab和模型的参数，用于重建模型，可以参考inference notebook

## 例子
- 写诗 （80 epochs）

```
标题: 秋思
正文: 秋风吹雨过，秋色满江城。一叶无人到，千山有客情。
标题: 百度
正文: 百尺孤城上，千金万里中。山川无限水，水石有余风。
标题: 湾区春日之谜
正文: 春风吹雨不成秋，春色如何一日休。不是春光无处着，只应春色是人愁。
```

- 对联 （60 epochs）

```
上: 欢天喜地度佳节
下: 举国迎春贺新年
上: 不待鸣钟已汗颜，重来试手竟何艰
下: 只缘沧海常风雨，再去翻身只等闲
上: 相思俱付三更月
下: 寂寞难留一夜风
```
