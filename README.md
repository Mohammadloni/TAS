# TAS: Ternarized Neural Architecture Search for Resource-Constrained Edge Devices
A PyTorch Implementation of the paper [https://www.es.mdh.se/pdf_publications/6351.pdf] (DATE 2022)

If you find any part of our code useful for your research, consider citing our paper.

```
@inproceedings{loni2022tas,
  title={TAS: Ternarized Neural Architecture Search for Resource-Constrained Edge Devices},
  author={Loni, Mohammad and Mousavi, Hamid and Riazati, Mohammad and Daneshtalab, Masoud and Sj{\"o}din, Mikael},
  booktitle={2022 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  pages={1115--1118},
  year={2022},
  organization={IEEE}
} 
```
# Authors
* Mohammad Loni [http://www.es.mdh.se/staff/3662-Mohammad_Loni]
* Hamid Mousavi [https://hamidmousavi0.github.io/]

# Intoduction 
We propose (i) a new cell template for ternary networks with a maximum gradient propagation; and (ii) a novel learnable quantizer that adaptively
relaxes the ternarization mechanism from the distribution of the weights and activation function.

# Searching Architectures
To search architectures, use the following command:
```
python src/search.py --save [ARCH_NAME]
```
*  [ARCH_NAME]: the name of the searched architecture

# Training Searched Architectures from Scratch
Train our best-searched cell on CIFAR10, using the following command:
```
python Src/train.py --learning_rate 0.05 --save [SAVE_NAME] --arch TER_ARCH_In_GENOTYPE --parallel 
```
Train our best-searched cell on ImageNet, use the following command:
```
python Src/train_imagenet.py --data [PATH_TO_IMAGENET] --arch TER_ARCH_In_GENOTYPE --model_config [MODEL_CONFIG] --save [SAVE_NAME]
```
# Convert To Keras 
The model is saved in PyTorch format "model.pt". To convert the model to Keras, you can use the following tools:
* [https://github.com/gmalivenko/pytorch2keras]
* [https://medium.com/analytics-vidhya/pytorch-to-keras-using-onnx-71d98258ad76]
# Contributors

Some of the code in this repository is based on the following amazing works:
* [https://github.com/gistvision/bnas]
