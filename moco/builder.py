# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        enable_asym_bn=False,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.encoder_q.fc,
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.encoder_k.fc,
        )

        """
# --------------------------------------------------------------------------- #
#                               Sync BatchNorm                                #
# --------------------------------------------------------------------------- #
Intermediate Sync BatchNorm layers is a way to reduce intra-image variance
intarget encoder. Sync BatchNorm leads to a notable improvement when applied to
target (as referred ‘AsymBN’ in our paper) and degeneration to source.
# --------------------------------------------------------------------------- #
        """

        if enable_asym_bn:
            process_group = create_syncbn_process_group(8)
            self.encoder_k.fc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.encoder_k.fc, process_group
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_q_mini, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_large = []
        for im in im_q:
            _q = self.encoder_q(im)  # queries: NxC
            _q = nn.functional.normalize(_q, dim=1)
            q_large.append(_q)

        q_mini = []
        for im in im_q_mini:
            _q_mini = self.encoder_q(im)  # queries: NxC
            _q_mini = nn.functional.normalize(_q_mini, dim=1)
            q_mini.append(_q_mini)

        """
# --------------------------------------------------------------------------- #
#                                 Mean Encoding                               #
# --------------------------------------------------------------------------- #
Mean Encoding is a direct approach to reduce the variance of a random variable
by performing i.i.d. sampling multiple times and take the mean as the new
variable. Mean Encoding is simply generated by running the same encoder on
multiple augmented views of the same image.
# --------------------------------------------------------------------------- #
        """

        crop_num = len(im_k)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k = torch.cat(im_k, dim=0)
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            cur_size, embedding_length = k.shape
            k = k.view(crop_num, cur_size // crop_num, embedding_length)
            k = nn.functional.normalize(torch.mean(k, dim=0), dim=1)

        logits_list = []
        labels_list = []
        for q in q_large + q_mini:
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            logits_list.append(logits)
            labels_list.append(labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits_list, labels_list


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def create_syncbn_process_group(num_gpu_per_group):
    if num_gpu_per_group == 0:
        return None

    world_size = torch.distributed.get_world_size()
    assert world_size >= num_gpu_per_group
    assert world_size % num_gpu_per_group == 0

    group = None
    for group_num in range(world_size // num_gpu_per_group):
        group_ids = range(
            group_num * num_gpu_per_group, (group_num + 1) * num_gpu_per_group
        )
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank() // num_gpu_per_group == group_num:
            group = cur_group

    assert group is not None
    return group
