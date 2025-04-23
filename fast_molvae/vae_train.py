import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import cPickle as pickle

from fast_jtnn import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print args

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
print model

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
scheduler.step()

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta = args.beta
meters = np.zeros(4)

import time  # 🕰️ for timing
import subprocess

for epoch in range(args.epoch):  # Python 3-compatible
    loader = MolTreeFolder(args.train, vocab, args.batch_size, num_workers=4)
    print(f"🚀 Starting epoch {epoch + 1} / {args.epoch} — Total step so far: {total_step}")
    epoch_start = time.time()

    for batch in loader:
        total_step += 1
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
        except Exception as e:
            print(f"❌ Step {total_step} — Exception during training: {e}")
            continue

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print(f"[{total_step}] ⏱️ Elapsed: {time.time() - epoch_start:.2f}s | "
                  f"Beta: {beta:.3f}, KL: {meters[0]:.2f}, Word: {meters[1]:.2f}, "
                  f"Topo: {meters[2]:.2f}, Assm: {meters[3]:.2f}, "
                  f"PNorm: {param_norm(model):.2f}, GNorm: {grad_norm(model):.2f}")
            sys.stdout.flush()
            meters *= 0

        if total_step % args.save_iter == 0:
          save_path = f"{args.save_dir}/model.iter-{total_step}"
          torch.save(model.state_dict(), save_path)
          print(f"💾 Saved checkpoint to {save_path}")

          # ✅ Optional: backup to Google Drive
          drive_dir = "/content/drive/MyDrive/jtnn_checkpoints"
          drive_path = f"{drive_dir}/model.iter-{total_step}"

          # Create the Drive folder if needed
          subprocess.run(["mkdir", "-p", drive_dir])
          subprocess.run(["cp", save_path, drive_path])

          print(f"📁 Backed up to Google Drive: {drive_path}")

        if total_step % args.anneal_iter == 0:
          scheduler.step()
          print(f"📉 Annealed LR to {scheduler.get_last_lr()[0]:.6f}")

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
          beta = min(args.max_beta, beta + args.step_beta)
