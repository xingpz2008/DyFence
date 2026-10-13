# Model Weight Provenance

The five CIFAR-10 checkpoints are the exact files used for Table IV. They are
stored with Git LFS under `model_weights/cifar10/` and checked by `ae.py`
before use.

| Model | File | SHA-256 |
|---|---|---|
| ResNet-18 | `resnet18_cifar10.pth` | `6d1c05b00288ac4f504a459e7ad2b95af4d7c00133c374872f4bc3cfe1a71659` |
| ResNet-50 | `resnet50_cifar10.pth` | `916212a4bb50fd12ccf49d404334bb68502d1c4f02bba07cb66600b21dbb69f4` |
| ResNet-101 | `resnet101_cifar10.pth` | `8b926ddf7c7a846457e742fd3c98d02bd21f07aa5d9197074e7f9863563e7aca` |
| VGG-11 BN | `vgg11_cifar10.pth` | `48180e333a9dcb883e897f1e6ad850ba54c0fea0e4dcfe298baa894848a8b015` |
| VGG-16 BN | `vgg16_cifar10.pth` | `96e61fbb1636ffc4c5125848efcd4a1bc5a6dbc93fd393c4eb027eac6a753aed` |

ImageNet uses the pretrained torchvision enums provided by the pinned
`torchvision==0.25.0+cpu` dependency.

| Model enum | Torchvision file | SHA-256 |
|---|---|---|
| `ResNet18_Weights.DEFAULT` | `resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| `ResNet50_Weights.DEFAULT` | `resnet50-11ad3fa6.pth` | `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| `ResNet101_Weights.DEFAULT` | `resnet101-cd907fc2.pth` | `cd907fc2a0de2292b2ba2b27d2216bc672d84632d1cf20058d01c0c03b5e020e` |
| `VGG11_BN_Weights.DEFAULT` | `vgg11_bn-6002323d.pth` | `6002323d9413ae02f657473d3ecbbd7f86ee496b8a0dc8dc6140a639a0fcf13d` |
| `VGG16_BN_Weights.DEFAULT` | `vgg16_bn-6c64b313.pth` | `6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a` |
