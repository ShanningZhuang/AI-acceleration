An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

Motivation:

卷积神经网络在计算机视觉任务中占据主导地位，但其局限性在于对局部特征的过度依赖，难以有效捕捉全局信息。Transformer架构在自然语言处理中取得了巨大成功，其自注意力机制能够有效捕捉长距离依赖关系。因此，作者尝试将Transformer架构应用于图像识别任务，以探索其在视觉领域的潜力。

Method:

首先对输入图像分块，将图像分割成固定大小的patch，进而可以将每个patch视为一个token以适配transformer结构。将处理后的图像token通过线性投影映射到高维空间，随后利用transformer结构中的位置编码、多头自注意力机制、前馈网络等机制捕捉所有token之间的长距离依赖关系。最终基于classification token进行图像分类。实验部分采用大规模数据集进行预训练，并在下游任务中进行微调。

Results:

在图像分类数据集上，Vision Transformer达到了与最先进的CNN模型相当或更优的性能，且训练速度快于ResNet。一个重要的实验观察是，Vision Transformer在大规模数据集上表现出色，但在小规模数据集上的性能不如CNN，表明其对训练数据量的需求较高，在大规模数据训练(scaling)条件下有潜在优势。在泛化性方面，Vision Transformer表现出良好的迁移学习能力，尤其是在大规模预训练条件下。

Limitations and possible future work:

Transformer结构的计算复杂度较高，尤其是其中的自注意力机制，未来可以探索更高效的注意力机制或稀疏化方法。

虽然Transformer擅长捕捉全局信息，但在局部、细粒度特征提取方面可能不如CNN，可以考虑结合CNN的优势，设计混合架构，并在需要局部信息提取的segmentation任务中进行应用。

   当前结果中，基于ViT的self-supervised训练相比supervised pre-training仍然差距较大，可以进一步探究能够更好适配ViT结构的自监督预训练算法。
