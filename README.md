# Ultra-Fast-Lane-Detection(ZJF)
PyTorch implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

**\[July 18, 2022\] Updates: The new version of our method has been accepted by TPAMI 2022. Code is available [here](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)**.

\[June 28, 2021\] Updates: we will release an extended version, which improves **6.3** points of F1 on CULane with the ResNet-18 backbone compared with the ECCV version.

Updates: Our paper has been accepted by ECCV2020.

![alt text](vis.jpg "vis")

The evaluation code is modified from [SCNN](https://github.com/XingangPan/SCNN) and [Tusimple Benchmark](https://github.com/TuSimple/tusimple-benchmark).

Caffe model and prototxt can be found [here](https://github.com/Jade999/caffe_lane_detection).

# Demo 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=lnFbAG3GBN4
" target="_blank"><img src="http://img.youtube.com/vi/lnFbAG3GBN4/0.jpg" 
alt="Demo" width="240" height="180" border="10" /></a>


# Install
Please see [INSTALL.md](./INSTALL.md)

# Get started
First of all, please modify `data_root` and `log_path` in your `configs/culane.py` or `configs/tusimple.py` config according to your environment. 
- `data_root` is the path of your CULane dataset or Tusimple dataset. 
- `log_path` is where tensorboard logs, trained models and code backup are stored. ***It should be placed outside of this project.***



***

For single gpu training, run
```Shell
python train.py configs/path_to_your_config
```
For multi-gpu training, run
```Shell
sh launch_training.sh
```
or
```Shell
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
```
If there is no pretrained torchvision model, multi-gpu training may result in multiple downloading. You can first download the corresponding models manually, and then restart the multi-gpu training.

Since our code has auto backup function which will copy all codes to the `log_path` according to the gitignore, additional temp file might also be copied if it is not filtered by gitignore, which may block the execution if the temp files are large. So you should keep the working directory clean.
***

Besides config style settings, we also support command line style one. You can override a setting like
```Shell
python train.py configs/path_to_your_config --batch_size 8
```
The ```batch_size``` will be set to 8 during training.

***

To visualize the log with tensorboard, run

```Shell
tensorboard --logdir log_path --bind_all
```

# Trained models
We provide two trained Res-18 models on CULane and Tusimple.

|  Dataset | Metric paper | Metric This repo | Avg FPS on GTX 1080Ti |    Model    |
|:--------:|:------------:|:----------------:|:-------------------:|:-----------:|
| Tusimple |     95.87    |       95.82      |         306         | [GoogleDrive](https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing)/[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA) |
|  CULane  |     68.4     |       69.7       |         324         | [GoogleDrive](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing)/[BaiduDrive(code:w9tw)](https://pan.baidu.com/s/19Ig0TrV8MfmFTyCvbSa4ag) |

# Evaluation and Visualization
**配置文件参数**
`model_mark` 标记 **模型所属的训练和在该训练中所属的批次** ，代码根据 `model_mark` 将测试结果和可视化结果分类放好，方便实验结束后分析。
`model_mark` = '模型所属的训练/在该训练中所属的批次'【即相对 `log_path` 的路径】

- `model_mark` is a mark represent the model you want to evaluate. 
- `test_model` is the full path of your evaluated model, which should be set as 'log_path' + `model_mark`


Configure as above, you can save evaluate or visualize results by category. You just need to **modify model_mark** when you evaluate or visualize the model, the code will automatically save the work in the place by category.

For evaluation

1、modify `model_mark`、`test_model` and `test_work_dir` in your `configs/culane.py` or `configs/tusimple.py` config according to your target.
- `test_work_dir` is the path to save your evaluated work, which should be set as 'path to save your evaluation' + `model_mark`

2、run
```Shell
python test.py configs/culane.py

python test.py configs/tusimple.py
```
代码会测试 `test_model` 并保存到 `test_work_dir`
Same as training, multi-gpu evaluation is also supported.

---
For visualization, firstly modify `model_mark`、`test_model` and `visualize_work_dir` in configuration file according to your target.
- `visualize_work_dir` is the path to save your visualized work, which should be set as 'path to save your visualization' + `model_mark`

We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane and Tusimple.
```Shell
python demo.py configs/culane.py
# or
python demo.py configs/tusimple.py
```
Since the testing set of Tusimple is not ordered, the visualized video might look bad and we **do not recommend** doing this.


运行可视化脚本，**代码会将 `test_model` 可视化并将结果保存到 `visualize_work_dir`**


# Speed
To test the runtime, please run
```Shell
python speed_simple.py  
# this will test the speed with a simple protocol and requires no additional dependencies

python speed_real.py
# this will test the speed with real video or camera input
```
It will loop 100 times and calculate the average runtime and fps in your environment.

# Citation

```BibTeX
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}

@ARTICLE{qin2022ultrav2,
  author={Qin, Zequn and Zhang, Pengyi and Li, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2022.3182097}
}
```

# Thanks
Thanks zchrissirhcz for the contribution to the compile tool of CULane, KopiSoftware for contributing to the speed test, and ustclbh for testing on the Windows platform.
