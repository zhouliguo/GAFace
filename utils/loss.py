# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        #self.L1Loss = nn.L1Loss()
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            if len(indices[i][0]):
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression
                    pxy = ps[:, :2]#.sigmoid() * 2. - 0.5
                    #pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pwh = torch.exp(ps[:, 2:4])#torch.pow(anchors[i], ps[:, 2:4].sigmoid() * 2)/anchors[i][0]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss
                    #lbox += self.L1Loss(pbox, tbox[i])

                    # Objectness
                    tobj[b, a, gj, gi] = 1#(1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                    # Classification
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain

            aw, ah = anchors[0]
            gi, gj = [], []
            gx, gy, gw, gh = [], [], [], []
            a, b, c = [], [], []
            #weights = (10., 10., 5., 5.)
            #weights = (1., 1., 0.5, 0.5)

            if nt:
                for ti in t[0]:
                    r = ti[4:6]*anchors[0]
                    r = torch.log(r)/torch.log(anchors[0])

                    if torch.max(r[0], r[1]) > 2 or torch.min(r[0], r[1]) < 0.5:
                        continue

                    gl = torch.round(ti[2]-ti[4]/2)
                    gr = torch.round(ti[2]+ti[4]/2)-1
                    gu = torch.round(ti[3]-ti[5]/2)
                    gd = torch.round(ti[3]+ti[5]/2)-1

                    if gr < gl:
                        gl = ti[2].int()
                        gr = ti[2].int()
                    if gd < gu:
                        gu = ti[3].int()
                        gd = ti[3].int()

                    gii = torch.arange(gl, gr+1).clamp_(0, gain[2] - 1).cuda()
                    gjj = torch.arange(gu, gd+1).clamp_(0, gain[3] - 1).cuda()
                    gii_len = gii.shape[0]
                    gii = gii.repeat(gjj.shape[0], 1).T.reshape(-1)
                    gjj = gjj.repeat(gii_len)

                    gi.append(gii)
                    gj.append(gjj)

                    gii_len = gii.shape[0]
                    gx.append(ti[2].repeat(gii_len) - gii - 0.5)
                    gy.append(ti[3].repeat(gii_len) - gjj - 0.5)
                    gw.append(ti[4].repeat(gii_len))
                    gh.append(ti[5].repeat(gii_len))
                    
                    a.append(ti[6].repeat(gii_len))
                    b.append(ti[0].repeat(gii_len))
                    c.append(ti[1].repeat(gii_len))

                if len(gi)>0 and len(gj)>0:
                    gi = torch.cat(gi, -1).long()
                    gj = torch.cat(gj, -1).long()
                    gx = torch.cat(gx, -1)
                    gy = torch.cat(gy, -1)
                    gw = torch.cat(gw, -1)
                    gh = torch.cat(gh, -1)
                    a = torch.cat(a, -1).long()
                    b = torch.cat(b, -1).long()
                    c = torch.cat(c, -1).long()
            else:
                return tcls, tbox, indices, anch
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            #tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            if len(gi)>0 and len(gj)>0:
                tbox.append(torch.stack((gx, gy, gw, gh)).T)  # box
            else:
                tbox.append((gx, gy, gw, gh))
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
