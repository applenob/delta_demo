# DELTA 使用pip的教程

## 简介

在本教程中，我们展示了使用**pip安装的用户**如何运行一个文本分类的任务。我们使用了一个简单的mock数据集作为演示。

一般一个完整的模型训练流程包括如下步骤：

- 准备数据集
- 开发自定义模块（可选）
- 设置配置文件
- 训练模型
- 导出模型

请先clone我们的demo仓库:

```bash
git clone --depth 1 https://github.com/applenob/delta_demo.git
cd ./delta_demo
```

## 安装的简单回顾

如果还没有安装 `delta-nlp`，请先安装:

```bash
pip install delta-nlp
```

**Requirements**: 需要 `tensorflow==2.0.0`
和`python==3.6`，系统是MacOS或者Linux.

## 准备数据集

```bash
./gen_data.sh
```

生成的数据在`data`目录。

生成的数据符合文本分类的标准格式，也就是"label\tdocument"。

## 开发自定义模块（可选）

在开发你自己的模块之前，请先确保我们的平台并没有这个模块。

```python
@registers.model.register
class TestHierarchicalAttentionModel(HierarchicalModel):
  """Hierarchical text classification model with attention."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)

    logging.info("Initialize HierarchicalAttentionModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']
    self.use_true_length = config['model'].get('use_true_length', False)
    if self.use_true_length:
      self.split_token = config['data']['split_token']
    self.padding_token = utils.PAD_IDX
```

在config文件`config/han-cls.yml`中注册模块文件（相对于运行环境的路径）：

```yml
custom_modules:
  - "test_model.py"
```

## 配置文件的配置

这里的配置文件是：`config/han-cls.yml`

在配置文件中，我们设置task是`TextClsTask`，模型是`TestHierarchicalAttentionModel`。

### 配置细节

配置文件主要由三部分构成：`data`, `model`, `solver`。
 
数据处理相关的配置在`data`字段下。你可以设置数据的路径（包括训练集、开发集和测试集）。
比如，我们设置`use_dense: false`，因为这里没有额外的dense输入。
我们设置`language: chinese`因为这是一个中文数据集。
 
模型相关的参数在`model`字段下。最重要的参数是`name:TestHierarchicalAttentionModel`，这个参数指定了我们使用的模型的类。
模型细节的参数一般存放在：`net->structure`。这里`max_sen_len`和`max_doc_len`都是32。

Solver相关的参数在`solver`字段下。包括优化器、评价的指标、和checkpoint的saver。这里使用的solver是`RawSolver`。

## 训练模型

配置好参数，就可以训练模型了：
```
delta --cmd train_and_eval --config config/han-cls.yml
```

参数 `cmd` 告诉平台边训练，边评估。
在训练了足够的轮数后，checkpoint会保存在配置`saver->model_path`指定的路径下，这里是exp/han-cls/ckpt`
in this case.

## 导出模型

如果你想导出一个特定的checkpoint，可以在配置文件中设置`infer_model_path`。
否则，平台会把`saver->model_path`目录下最新的checkpoint导出。

```
delta --cmd export_model --config/han-cls.yml
```

导出的模型存放在路径：`service->model_path`下，这里是 `exp/han-cls/service`。
here.
