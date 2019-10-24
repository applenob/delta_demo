# DELTA 使用pip的教程

## 安装

目前还未上传pypi，只提供mac环境下nlp版支持。

在**已经安装tensorflow2.0.0**的环境执行：

```bash
pip install wheel_houses/delta_didi-0.2-cp36-cp36m-macosx_10_7_x86_64.whl
```

## 生成数据

```bash
./run.sh
```

## 编写自定义模块（可选）

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

## 运行

```bash
delta --cmd train_and_eval --config config/han-cls.yml
```
