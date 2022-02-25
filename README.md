# chinese-ai-writing-share
中文AI写作分享

个人非常满意的个别案例
```
#### Mengzi T5 Finetune ####
上： 不待鸣钟已汗颜，重来试手竟何艰
下： 何堪击鼓频催泪？一别伤心更枉然
上： 北国风光，千里冰封，万里雪飘
下： 南疆气象，五湖浪涌，三江潮来

標題： 作诗：中秋
詩歌： 秋氣侵肌骨，寒光入鬢毛。雲收千里月，風送一帆高。
標題： 作诗：中秋 模仿：苏轼
詩歌： 月從海上生，照我庭下影。不知此何夕，但見天宇靜。

#### transformer supervised learning ####
上: 不待鸣钟已汗颜，重来试手竟何艰
下: 只缘沧海常风雨，再去翻身只等闲
上: 相思俱付三更月
下: 寂寞难留一夜风

标题: 秋思
正文: 秋风吹雨过，秋色满江城。一叶无人到，千山有客情。
标题: 湾区春日之谜
正文: 春风吹雨不成秋，春色如何一日休。不是春光无处着，只应春色是人愁。
```

## 声明
- 基于主流transformer的业余水平语言模型实现，几小时见效，适合自娱自乐
- 数据质量直接影响模型效果，模型也因为个人计算资源有限没有压榨性能极限，如模型有异常或低质量输出，请大家见谅
- 2021 keras transformer Demo使用 `topk=1`
- 2022 T5 诗歌使用默认设置 `num_beams=2, top_k=50, top_p=0.95`，对联使用`topk=1`

## 架构
- 2021监督学习方案，自己从头训练
    - 基于Transformer的encoder-decoder
    - transformer使用keras-transformer lib
- 2022迁移学习方案，使用T5 finetune
    - 预训练使用 [澜舟科技的孟子 T5](https://huggingface.co/Langboat/mengzi-t5-base)
    - 理论上可以把诗歌和对联两个合起来作为multi-task下游任务，但是对联有很多是现代白话文，古文我只用了唐诗宋词，所以最后还是分开
    - 我只训练了3-4个epoch，看loss的下降速度应该还有很大提升空间

## 数据来源
- 唐诗宋词 https://github.com/chinese-poetry/chinese-poetry
  - 2021 transformer 只训练 `标题 -> 诗歌`
  - 2022 T5 方案考虑了 `标题 -> 诗歌`，或者 `标题+诗人 -> 诗歌`
  - 标题长度限制12token，诗人4token，诗歌64token，结尾用句号，具体参考training下面的notebook
- 对联 https://github.com/wb14123/couplet-dataset
  - 标准输入输出，T5使用`对联：`前缀，长度限制32字符

## 语言支持
- 默认简体中文
- 2022 T5 inference 支持繁体中文，需要标记 `is_input_traditional_chinese=True`
- 如需要训练繁体中文模型，查找`chinese_converter.to_simplified`改为`chinese_converter.to_traditional`

## 训练
- 我是用 Google Colab Pro（推荐，16G的GPU一个月随便用才9.99！）
- transformer方案使用TF2 keras，用TPU训练，模型训练时间~10小时
- T5因为使用simplet5 (pytorch + huggingface 的一个封装)，所以使用GPU训练，模型训练时间~6-8小时

## 模型下载和使用
- 推荐参考inference下面的notebook来参考使用，模型下载地址也在notebook介绍
  - **重要：模型文件存在Google Drive，推荐用Google账号打开，点击`Add to shortcut`，之后在你Drive的主页面`shared with me`看到目录后选择`add shortcut to Drive`，这样可以mount后本地可以操作文件**
- 模型参数大小
  - 2021 Transformer 对联 ~80M
  - 2021 Transformer 写诗 ~10M
  - 2022 T5 ~250M

## 更多例子，也可以参考inference下面的notebook
- 写诗 2021 transformer 方案示例 （80 epochs）

```
标题: 秋思
正文: 秋风吹雨过，秋色满江城。一叶无人到，千山有客情。
标题: 百度
正文: 百尺孤城上，千金万里中。山川无限水，水石有余风。
标题: 湾区春日之谜
正文: 春风吹雨不成秋，春色如何一日休。不是春光无处着，只应春色是人愁。
```

- 写诗 2022 T5 finetune 方案示例（4 epochs）

```
标题： 作诗：秋思
诗歌： 秋思不可奈，况复值新晴。露叶红犹湿，风枝翠欲倾。客愁随日薄，归夢逐云轻。独倚阑干久，西风吹雁声。
标题： 作诗：秋思 模仿：杜甫
诗歌： 西风动高树，落叶满空庭。白露侵肌冷，青灯照眼青。客愁随暮角，归夢逐残星。独坐还成感，秋声不可听。
标题： 作诗：秋思 模仿：李白
诗歌： 秋色满空山，秋风动客衣。浮云不到处，明月自来归。
标题： 作诗：秋思 模仿：李清照
诗歌： 秋思不可奈，况复在天涯。客路逢寒食，家书报早炊。风霜侵鬓发，天地入诗脾。欲寄南飞雁，归期未有期。
标题： 作诗：秋思 模仿：苏轼
诗歌： 西风吹雨过江城，独倚阑干思不胜。黄叶满庭秋意动，碧梧当户夜寒生。故园夢断人千里，新雁书来雁一行。莫怪衰翁无业，一樽聊复慰平生。

标题： 作诗：百花
诗歌： 百花开尽绿阴成，红紫妖红照眼明。谁道东风无意思，一枝春色爲谁荣。
标题： 作诗：百花 模仿：杜甫
诗歌： 百花开尽绿阴成，独有江梅照眼明。莫道春光无别意，只应留得一枝横。
标题： 作诗：百花 模仿：李白
诗歌： 百花如锦树，春色满芳洲。日暖花争发，风轻絮乱流。香飘金谷露，艳拂玉山楼。谁道无情物，年年爲客愁。
标题： 作诗：百花 模仿：李清照
诗歌： 百花如锦水如蓝，春到园林处处堪。谁道东风不相识，一枝开尽绿阴南。
标题： 作诗：百花 模仿：苏轼
诗歌： 百花开尽绿阴成，谁道春风不世情。若使此花无俗韵，世间那得有芳名。

標題： 作诗：春节
詩歌： 去年今日到江干，家在青山綠水間。老去心情渾似舊，春來情緒只如閒。
標題： 作诗：春节 模仿：杜甫
詩歌： 江上春歸早，山中客到稀。亂花隨處發，細草向人飛。節物催年老，生涯逐日非。故園桃李樹，猶得及芳菲。
標題： 作诗：春节 模仿：李白
詩歌： 去年今日來，花發滿城開。今歲明朝去，明年依舊來。
標題： 作诗：春节 模仿：李清照
詩歌： 去年今日是今朝，不覺今年又一宵。但有梅花堪共醉，何須柳絮更相撩。
標題： 作诗：春节 模仿：苏轼
詩歌： 今年春色到江干，柳眼桃腮次第看。但得此身長健在，不須回首歎凋殘。

標題： 作诗：中秋
詩歌： 秋氣侵肌骨，寒光入鬢毛。雲收千里月，風送一帆高。
標題： 作诗：中秋 模仿：杜甫
詩歌： 秋色滿江天，清光萬里懸。雲開見海月，水落見沙田。白露侵肌冷，青苔滿鬢鮮。何當一樽酒，共醉玉壺前。
標題： 作诗：中秋 模仿：李白
詩歌： 中秋月色好，況復是中秋。玉兔擣藥杵，金烏搗藥。雲開天似水，風起海如漚。此夜何人見，長歌淚不流。
標題： 作诗：中秋 模仿：李清照
詩歌： 秋氣侵肌骨，寒光入鬢毛。客愁隨日減，詩思逐風高。露重衣襟溼，天高雁影豪。何當一樽酒，來此醉陶陶。
標題： 作诗：中秋 模仿：苏轼
詩歌： 月從海上生，照我庭下影。不知此何夕，但見天宇靜。
```

- 对联 2021 transformer 方案示例 （60 epochs）

```
上: 欢天喜地度佳节
下: 举国迎春贺新年
上: 不待鸣钟已汗颜，重来试手竟何艰
下: 只缘沧海常风雨，再去翻身只等闲
上: 相思俱付三更月
下: 寂寞难留一夜风
```

- 对联 2022 T5 finetune 方案示例（3 epochs）
```
上： 欢天喜地度佳节
下： 笑语欢歌迎新春
上： 不待鸣钟已汗颜，重来试手竟何艰
下： 何堪击鼓频催泪?一别伤心更枉然
上： 当年欲跃龙门去，今日真披马革还
下： 此日当登虎榜来，他年又见龙图新
上： 载歌在谷
下： 对酒当歌
上： 北国风光，千里冰封，万里雪飘
下： 南疆气象，五湖浪涌，三江潮来
上： 寂寞寒窗空守寡
下： 逍遥野渡醉吟诗
上： 烟锁池塘柳
下： 云封岭上松
上： 五科五状元，金木水火土
下： 三才三进士，诗书礼乐诗
上： 望江楼，望江流，望江楼上望江流，江楼千古，江流千古
下： 听雨阁，听雨落，听雨阁中听雨落，雨阁万重，雨落万重
上： 載歌在谷
下： 對酒當歌
上： 飛龍在天
下： 臥虎於淵
上： 都說臺北風光好
下： 不曉臺灣景色新
```
