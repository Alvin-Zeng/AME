import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Combination(nn.Module):
    def __init__(self, model_s: nn.Module, model_f: nn.Module, shared_layers: int):
        """
        :param model_s: slow model
        :param model_f: fast model
        :param shared_layers: the last shared layer (start at 0)
        """
        super(Combination, self).__init__()
        self.model_s = model_s
        self.model_f = model_f
        self.shared_layers = shared_layers
        slow_backbone = nn.Sequential(*list(self.model_s.children()))[0]
        fast_backbone = nn.Sequential(*list(self.model_f.children()))[0]
        self.cls_head_slow = nn.Sequential(*list(self.model_s.children()))[1]
        self.cls_head_fast = nn.Sequential(*list(self.model_f.children()))[1]
        self.slow_deep = nn.Sequential(*list(slow_backbone.children()))[self.shared_layers:]
        self.fast_deep = nn.Sequential(*list(fast_backbone.children()))[self.shared_layers:]
        self.backbone_shallow = nn.Sequential(*list(slow_backbone.children()))[:self.shared_layers]

    def forward(self, x, return_slow=True, return_feats=False, both=False, alignment=False):
        """
        :param alignment:
        :param x: trainmode[B,2(slow,fast),frames,3,224,224] else[B,1,frames,3,224,224]
        :param return_slow:
        :param return_feats:
        :param both:
        :return:
        """
        if both:
            batch_size = x.shape[0]
            num_frames = x.shape[2]
            img_shape = x.shape[-3:]
            x = x.reshape(-1, *img_shape)

            x = self.backbone_shallow(x)  # [B*2*frames,...]
            feature_shape = x.shape[1:]
            x = x.reshape(batch_size, 2, num_frames, *feature_shape)
            x_slow = x[:, 0].reshape(-1, *feature_shape)  # [B*frames,...]
            x_fast = x[:, 1].reshape(-1, *feature_shape)
            del x

            if alignment:
                means_slow = list()
                vars_slow = list()
                means_fast = list()
                vars_fast = list()

                alignment_indices = [1]
                i = 0
                for b in self.slow_deep.children():
                    if i in alignment_indices:
                        for l in b.children():
                            x_slow, mean, var = l(x_slow, statistic=True)
                            means_slow.extend(mean)
                            vars_slow.extend(var)
                    else:
                        for l in b.children():
                            x_slow = l(x_slow, statistic=False)
                    i = i + 1
                x_slow = torch.flatten(x_slow, 1)
                x_slow, logits_slow = self.cls_head_slow(x_slow)

                i = 0
                for b in self.fast_deep.children():
                    if i in alignment_indices:
                        for l in b.children():
                            x_fast, mean, var = l(x_fast, statistic=True)
                            means_fast.extend(mean)
                            vars_fast.extend(var)
                    else:
                        for l in b.children():
                            x_fast = l(x_fast, statistic=False)
                    i = i + 1
                x_fast = torch.flatten(x_fast, 1)
                x_fast, logits_fast = self.cls_head_slow(x_fast)

                for i, (ms, vs, mf, vf) in enumerate(zip(means_slow, vars_slow, means_fast, vars_fast)):
                    means_slow[i] = ms.view(1, -1).cuda()
                    vars_slow[i] = vs.view(1, -1).cuda()
                    means_fast[i] = mf.view(1, -1).cuda()
                    vars_fast[i] = vf.view(1, -1).cuda()

                return x_slow, logits_slow, x_fast, logits_fast, means_slow, vars_slow, means_fast, vars_fast

            else:
                x_slow = self.slow_deep(x_slow)
                x_slow = torch.flatten(x_slow, 1)
                x_slow, logits_slow = self.cls_head_slow(x_slow)
                x_fast = self.fast_deep(x_fast)
                x_fast = torch.flatten(x_fast, 1)
                x_fast, logits_fast = self.cls_head_fast(x_fast)

                return x_slow, logits_slow, x_fast, logits_fast

        else:
            img_shape = x.shape[-3:]
            x = x.reshape(-1, *img_shape)

            x = self.backbone_shallow(x)

            if return_slow:
                x = self.slow_deep(x)
                x = torch.flatten(x, 1)
                feat, logits = self.cls_head_slow(x)
            else:
                x = self.fast_deep(x)
                x = torch.flatten(x, 1)
                feat, logits = self.cls_head_fast(x)

            if return_feats:
                return feat, logits
            return logits

    @property
    def num_classes(self):
        return self.model_s.num_classes

    @property
    def output_dim(self):
        return self.model_s.output_dim

    def get_optim_policies(self, base_lr, deep_mult):
        shared_layers_params = {'params': self.backbone_shallow.parameters()}
        slow_deep_params = {'params': self.slow_deep.parameters()}
        fast_deep_params = {'params': self.fast_deep.parameters()}
        if deep_mult != 1:
            slow_deep_params['lr'] = base_lr * deep_mult
            fast_deep_params['lr'] = base_lr * deep_mult
        cls_head_slow_params = {'params': self.cls_head_slow.parameters()}
        cls_head_fast_params = {'params': self.cls_head_fast.parameters()}
        return [shared_layers_params, slow_deep_params, fast_deep_params, cls_head_slow_params, cls_head_fast_params]


class CombinationMoco(nn.Module):
    def __init__(self, src_model, momentum_model, K=16384, m=0.999, T_moco=0.07, checkpoint_path=None):
        """
        :param src_model:
        :param momentum_model:
        :param K: buffer size; number of keys
        :param m: moco momentum of updating key encoder (default: 0.999)
        :param T_moco: softmax temperature (default: 0.07)
        :param checkpoint_path:
        """
        super(CombinationMoco, self).__init__()

        self.K = K
        self.m = m
        self.T_moco = T_moco
        self.queue_ptr = 0

        # create the encoders
        self.src_model = src_model
        self.momentum_model = momentum_model

        # create the fc heads
        feature_dim = src_model.output_dim

        # freeze key model
        self.momentum_model.requires_grad_(False)

        # create the memory bank
        self.register_buffer("mem_feat", torch.randn(feature_dim, K))
        self.register_buffer( "mem_labels", torch.randint(0, src_model.num_classes, (K,)))
        self.mem_feat = F.normalize(self.mem_feat, dim=0)

        self.queue_is_full = False

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # encoder_q -> encoder_k
        for param_q, param_k in zip(self.src_model.parameters(), self.momentum_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels):
        """
        Update features and corresponding pseudo labels
        """
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.queue_ptr = end % self.K

    def forward(self, im_q, im_k=None, return_slow=True, cls_only=False, both=False, alignment=False):
        """
        :param alignment:
        :param im_q: a batch of query images [batchsize, sample_times=1, frames, 3, 224, 224]
        :param im_k: a batch of key images
        :param return_slow:
        :param cls_only:
        :param both:
        :return:
        """
        if both:
            if alignment:  # align the slow and fast
                batch_size = im_q.shape[0]
                x_slow, logits_slow, x_fast, logits_fast, means_slow, vars_slow, means_fast, vars_fast = self.src_model(im_q, both=True, alignment=True)

                q_slow = F.normalize(x_slow, dim=1)
                q_slow = q_slow.view(batch_size, -1)  # [B,features]
                q_fast = F.normalize(x_fast, dim=1)
                q_fast = q_fast.view(batch_size, -1)  # [B,features]

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder()  # update the key encoder

                    batch_size = im_k.shape[0]
                    frame_shape = im_k.shape[-4:]
                    k_slow = im_k[:, 0].reshape(batch_size, 1, *frame_shape)
                    k_fast = im_k[:, 1].reshape(batch_size, 1, *frame_shape)

                    k_slow, _ = self.momentum_model(k_slow, return_slow=True, return_feats=True, both=False)
                    k_slow = F.normalize(k_slow, dim=1)
                    k_fast, _ = self.momentum_model(k_fast, return_slow=True, return_feats=True, both=False)
                    k_fast = F.normalize(k_fast, dim=1)

                # compute logits
                l_pos_slow = torch.einsum('nc,nc->n', [q_slow, k_slow]).unsqueeze(-1)
                l_neg_slow = torch.einsum("nc,ck->nk", [q_slow, self.mem_feat.clone().detach()])
                l_pos_fast = torch.einsum('nc,nc->n', [q_fast, k_fast]).unsqueeze(-1)
                l_neg_fast = torch.einsum("nc,ck->nk", [q_fast, self.mem_feat.clone().detach()])

                # logits: N,(1+K)
                l_slow = torch.cat([l_pos_slow, l_neg_slow], dim=1)
                l_fast = torch.cat([l_pos_fast, l_neg_fast], dim=1)

                # apply temperature
                l_slow /= self.T_moco
                l_fast /= self.T_moco

                return x_slow, logits_slow, x_fast, logits_fast, k_slow, k_fast, l_slow, l_fast, means_slow, vars_slow, means_fast, vars_fast

            else:
                batch_size = im_q.shape[0]
                x_slow, logits_slow, x_fast, logits_fast = self.src_model(im_q, both=True)
                if cls_only:
                    return x_slow, logits_slow, x_fast, logits_fast

                q_slow = F.normalize(x_slow, dim=1)
                q_slow = q_slow.view(batch_size, -1)  # [B,features]
                q_fast = F.normalize(x_fast, dim=1)
                q_fast = q_fast.view(batch_size, -1)  # [B,features]

                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder()  # update the key encoder

                    batch_size = im_k.shape[0]
                    frame_shape = im_k.shape[-4:]
                    k_slow = im_k[:, 0].reshape(batch_size, 1, *frame_shape)
                    k_fast = im_k[:, 1].reshape(batch_size, 1, *frame_shape)

                    # shuffle for making use of BN
                    k_slow, idx_unshuffle_slow = self._batch_shuffle_ddp(k_slow)
                    k_fast, idx_unshuffle_fast = self._batch_shuffle_ddp(k_fast)

                    k_slow, _ = self.momentum_model(k_slow, return_slow=True, return_feats=True, both=False)
                    k_slow = F.normalize(k_slow, dim=1)
                    k_fast, _ = self.momentum_model(k_fast, return_slow=True, return_feats=True, both=False)
                    k_fast = F.normalize(k_fast, dim=1)

                    # undo shuffle
                    k_slow = self._batch_unshuffle_ddp(k_slow, idx_unshuffle_slow)
                    k_fast = self._batch_unshuffle_ddp(k_fast, idx_unshuffle_fast)

                l_pos_slow = torch.einsum('nc,nc->n', [q_slow, k_slow]).unsqueeze(-1)
                l_neg_slow = torch.einsum("nc,ck->nk", [q_slow, self.mem_feat.clone().detach()])
                l_pos_fast = torch.einsum('nc,nc->n', [q_fast, k_fast]).unsqueeze(-1)
                l_neg_fast = torch.einsum("nc,ck->nk", [q_fast, self.mem_feat.clone().detach()])

                # logits: N,(1+K)
                l_slow = torch.cat([l_pos_slow, l_neg_slow], dim=1)
                l_fast = torch.cat([l_pos_fast, l_neg_fast], dim=1)

                # apply temperature
                l_slow /= self.T_moco
                l_fast /= self.T_moco

                return x_slow, logits_slow, x_fast, logits_fast, k_slow, k_fast, l_slow, l_fast

        else:
            # compute query features
            feats_q, logits_q = self.src_model(im_q, return_slow=return_slow, return_feats=True)

            if cls_only:
                return feats_q, logits_q

            q = F.normalize(feats_q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k, _ = self.momentum_model(im_k, return_feats=True)
                k = F.normalize(k, dim=1)

            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            l_neg = torch.einsum("nc,ck->nk", [q, self.mem_feat.clone().detach()])
            logits_ins = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
            logits_ins /= self.T_moco  # apply temperature

            return feats_q, logits_q, logits_ins, k


