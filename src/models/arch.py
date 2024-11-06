#!/usr/bin/env python3
import torch
from torch import nn
import numpy as np
import math
from src import const


# @title ViT Implementation


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in
    Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from
    https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi)
                              * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) ->
        # (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of
        # the input sequence and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and
        # the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = nn.Parameter(
            torch.randn(
                1,
                self.patch_embeddings.num_patches + 1,
                config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) ->
        # (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if (c == 1):
            attention_probs = attention_probs + \
                torch.randn(attention_probs.size()).to(const.device) * \
                const.std + const.mean
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of
        # attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection
        # layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to
        # the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(
            self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, c, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x, c) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs],
            dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities
        # (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs],
                dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query,
    key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by
        # the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection
        # layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(
            self.hidden_size,
            self.all_head_size * 3,
            bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to
        # the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(
            self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, c, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) ->
        # (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) ->
        # (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads,
        # sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size).transpose(
            1,
            2)
        key = key.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size).transpose(
            1,
            2)
        value = value.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size).transpose(
            1,
            2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if (c == 1):
            attention_probs = attention_probs + \
                torch.randn(attention_probs.size()).to(const.device) * \
                const.std + const.mean
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length,
        # attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(
            1, 2) .contiguous() .view(
            batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities
        # (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(
            config["hidden_size"],
            config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(
            config["intermediate_size"],
            config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, c, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), c, output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities
        # (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config, num_encoders):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(num_encoders):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, c, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(
                x, c, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities
        # (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.num_encoders = config["num_hidden_layers"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config, self.num_encoders)
        # Create a linear layer to project the encoder's output to the number
        # of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, c, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(
            embedding_output, c, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for
        # classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)


class ViTForClassfication_g(nn.Module):
    """
    This model consists of the embedding layer and the first two encoder blocks
    of the encoder stack. It takes an input with dimensions
    batch x channel x height x width, and outputs features
    with dimensions batch x (patches + 1) x hidden_size.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.num_encoders = 2
        self.encoder = Encoder(config, self.num_encoders)
        # Create a linear layer to project the encoder's output to the number
        # of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, c, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(
            embedding_output, c, output_attentions=output_attentions)

        if not output_attentions:
            return (encoder_output, None)
        else:
            return (encoder_output, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)


class ViTForClassfication_f(nn.Module):
    """
    This model includes the remaining encoder blocks and the MLP
    head. It takes inputs with dimensions batch x (patches + 1)
    x latent_size and produce an output of dimensions
    batch x 1, representing the final classification prediction.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.num_encoders = 2
        self.encoder = Encoder(config, self.num_encoders)
        # Create a linear layer to project the encoder's output to the number
        # of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, c, output_attentions=False):
        # Calculate the embedding output

        encoder_output, all_attentions = self.encoder(
            x, c, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for
        # classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)


def select_exemplars(X, m):
    """
        Incremental exemplar selection algorithm.
        Args:
        - X: Set of input images as PyTorch tensors.
        - m: Target number of exemplars.
        Returns:
        - P: Set of exemplars as PyTorch tensors.
    """
    # Initialize the class mean
    mu = torch.mean(torch.stack(X), dim=0)
    # Initialize the exemplar set
    P = []
    # Exemplar selection loop
    for k in range(1, m + 1):
        # Compute distances and select exemplar
        if not P:
            distances = [torch.norm(mu - (1 / k) * (x)) for x in X]
        else:
            distances = [torch.norm(
                mu - (1 / k) * ((x) + torch.sum(torch.stack(P), dim=0)))
                for x in X]
        p_k_index = torch.argmin(torch.stack(distances))
        p_k = X[p_k_index]
        X.pop(p_k_index)
        # Update class mean with the new exemplar
        mu = (mu * (k - 1) + p_k) / k
        # Add exemplar to the set
        P.append(p_k)
    # Convert the list of tensors to a single tensor before returning
    return P


# defining class for episodic memory
class D_buffer:
    """
        Args:
            max_length - max number of images stored in the buffer
            tasks - the number of maximum tasks whose representations can be
            stored in the buffer
            num_classes - the total number of classes in the dataset
            batch_size - batch size of the training dataset
    """

    def __init__(self, max_length, batch_size, num_classes, tasks):
        self.max_length = max_length
        self.buffer_images = []
        self.buffer_labels = []
        self.num_elements = 0
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.tasks = tasks
        # list to store images based on class
        self.class_seperate_list = {i: [] for i in range(num_classes)}
        # list to store images based on icarl ranking for each class
        self.class_icarl_list = {}

    """
        Args:
            task_sem_mem_list - images from the current task
            that need to be stored in the memory
            task_num - current task number
    """

    def update(self, task_sem_mem_list, task_num):
        """
            update using icarl herding
            seperate batches into seperate seperate images and labels list
            seperate images based on labels
            implement icarl herding on each labelset
            take first max_length / (task_num * num_classes / tasks)
        """

        num_needed_per_class = int(
            self.max_length / (((task_num + 1) * const.num_classes) /
                               self.tasks))
        self.buffer_images = []
        self.buffer_labels = []
        images_list = [tup[0] for tup in task_sem_mem_list]
        labels_list = [tup[1] for tup in task_sem_mem_list]
        concatenate_images = torch.cat(images_list, dim=0)
        concatenate_labels = torch.cat(labels_list, dim=0)
        single_images_list = torch.split(concatenate_images, 1, dim=0)
        single_images_list = [
            tensor.squeeze(
                dim=0) for tensor in single_images_list]
        single_labels_list = torch.split(concatenate_labels, 1, dim=0)
        single_labels_list = [
            tensor.squeeze(
                dim=0).item() for tensor in single_labels_list]
        task_set = set(single_labels_list)
        for i in range(len(single_images_list)):
            self.class_seperate_list[int(single_labels_list[i])].append(
                single_images_list[i])
        for i in task_set:
            print(f'length of {i}')
            print(len(self.class_seperate_list[int(i)]))
            select_exemplar_length = min(
                self.max_length, len(self.class_seperate_list[int(i)]))
            self.class_icarl_list[i] = select_exemplars(
                self.class_seperate_list[int(i)], select_exemplar_length)
            print('done')

        for i in range(const.num_classes):
            if (i in self.class_icarl_list):
                self.buffer_images.extend(
                    self.class_icarl_list[i][0:num_needed_per_class])
                self.buffer_labels.extend([i] * num_needed_per_class)

        self.num_elements = len(self.buffer_images)
        return

    """
        Function which returns a batch of representations
        choosen at random
    """

    def get_batch(self):
        batch_images = []
        batch_labels = []
        for i in range(self.batch_size):
            index = np.random.randint(0, self.num_elements)
            batch_images.append(self.buffer_images[index])
            batch_labels.append(torch.tensor(self.buffer_labels[index]))
        images = torch.stack(batch_images, dim=0)
        labels = torch.stack(batch_labels, dim=0)
        return images, labels

    """
        Function to print all the images in the episodic memory
    """

    def print(self):
        for item in self.buffer_images:
            print(item)
    """
        Function which checks if the episodic memory (buffer)
        is empty
    """

    def is_empty(self):
        return (len(self.buffer_images) == 0)


# create model instances for training and defining optimizer and loss function
def get_models():
    model_g = ViTForClassfication_g(const.config).to(const.device)
    model_f_w = ViTForClassfication_f(const.config).to(const.device)
    model_f_s = ViTForClassfication_f(const.config).to(const.device)
    return model_g, model_f_w, model_f_s
