import argparse, os, re, random, shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch, torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

AMP = True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def human(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def stem_from_basename(b: str) -> str:
    b = re.sub(r"\.nii(\.gz)?$", "", b, flags=re.I)
    b = re.sub(r"\.(img|label)$", "", b, flags=re.I)
    b = re.sub(r"_(img|image|label|mask|seg|annotation|gt|groundtruth)$", "", b, flags=re.I)
    return b


def list_recursive(root: Path, pattern: str) -> List[str]:
    return [str(p) for p in root.rglob(pattern)]


class SliceDS(Dataset):

    def __init__(self, frame, tfm=None):
        import pandas as _pd
        self.df = frame.reset_index(drop=True) if isinstance(frame, _pd.DataFrame) else frame
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        ip = self.df.loc[i, "image"]
        mp = self.df.loc[i, "mask"]

        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {ip}")
        img = img.astype("float32") / 255.0

        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise FileNotFoundError(f"Mask not found: {mp}")
        msk = (msk > 127).astype("float32")

        img = img[..., None]
        msk = msk[..., None]

        if self.tfm:
            out = self.tfm(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        if img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1).contiguous()
        if msk.ndim == 3 and msk.shape[0] != 1:
            msk = msk.permute(2, 0, 1).contiguous()

        return img, msk


def cmd_extract(zip_path: Path, out_dir: Path):
    ensure_dir(out_dir)
    import zipfile
    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print("Extraction complete")


def find_pairs(root: Path) -> List[Tuple[str, str]]:

    img_patterns = ["*.img.nii.gz", "*_img.nii.gz", "*image*.nii.gz"]
    lbl_patterns = ["*.label.nii.gz", "*_label.nii.gz", "*mask*.nii.gz", "*seg*.nii.gz",
                    "*annotation*.nii.gz", "*gt*.nii.gz", "*ground*.nii.gz"]

    imgs = []
    for p in img_patterns:
        imgs += list_recursive(root, p)
    imgs = sorted(set(imgs))

    lbls = []
    for p in lbl_patterns:
        lbls += list_recursive(root, p)
    lbls = sorted(set(lbls))

    img_map, lbl_map = {}, {}
    for ip in imgs:
        img_map.setdefault(stem_from_basename(Path(ip).name), []).append(ip)
    for lp in lbls:
        lbl_map.setdefault(stem_from_basename(Path(lp).name), []).append(lp)

    pairs = []
    for k, ips in img_map.items():
        if k in lbl_map:
            pairs.append((ips[0], lbl_map[k][0]))

    return pairs


def cmd_slice(in_dir: Path, out_dir: Path, chunk_size: int = 50):

    ensure_dir(out_dir / "images")
    ensure_dir(out_dir / "masks")
    state_file = out_dir / "processed_volumes.txt"

    processed = set()
    if state_file.exists():
        processed = set([ln.strip() for ln in state_file.read_text(encoding="utf-8").splitlines() if ln.strip()])

    pairs = find_pairs(in_dir)
    print(f"Found {human(len(pairs))} image/label pairs")

    import nibabel as nib

    done = 0
    new_slices = 0
    for (img_path, lbl_path) in tqdm(pairs, desc="Slicing", unit="vol"):
        pid = re.sub(r"\.img\.nii\.gz$", "", Path(img_path).name)
        if pid in processed:
            continue

        try:
            img_vol = nib.load(img_path).get_fdata()
            lbl_vol = nib.load(lbl_path).get_fdata()

            created = 0
            for z in range(img_vol.shape[2]):
                img_slice = img_vol[:, :, z]
                lbl_slice = lbl_vol[:, :, z]
                if lbl_slice.max() == 0:
                    continue

                mn, mx = float(img_slice.min()), float(img_slice.max())
                norm = (img_slice - mn) / (mx - mn + 1e-8)
                img_u8 = (norm * 255).astype(np.uint8)
                msk_u8 = (lbl_slice > 0).astype(np.uint8) * 255

                name = f"{pid}_slice_{z:03d}.png"
                cv2.imwrite(str(out_dir / "images" / name), img_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(str(out_dir / "masks" / name), msk_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                created += 1
                new_slices += 1

            with state_file.open("a", encoding="utf-8") as f:
                f.write(pid + "\n")
            processed.add(pid)
            done += 1

            if done >= chunk_size:
                break

        except Exception as e:
            print(f"  ⚠️ Error {pid}: {e}")

    print("\n=== CHUNK SUMMARY ===")
    print(f"processed this run: {done}")
    print(f"new slices created: {human(new_slices)}")
    remaining = sum(1 for (ip, _) in pairs if re.sub(r'\.img\.nii\.gz$', '', Path(ip).name) not in processed)
    print(f"remaining volumes:  {human(remaining)}")
    print(f"slices saved to:    {out_dir}")


def cmd_make_csv(slices_dir: Path):

    images_root, masks_root = slices_dir / "images", slices_dir / "masks"
    exts = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")

    img_paths, msk_paths = [], []
    for ext in exts:
        img_paths += [str(p) for p in images_root.rglob(f"*.{ext}")]
        msk_paths += [str(p) for p in masks_root.rglob(f"*.{ext}")]

    def base_noext(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    imap = {base_noext(p): p for p in img_paths}
    mmap = {base_noext(p): p for p in msk_paths}
    common = sorted(set(imap) & set(mmap))

    if not common:
        raise SystemExit("No matching image/mask filenames found under images/ and masks/.")

    df = pd.DataFrame({"image": [imap[k] for k in common],
                       "mask": [mmap[k] for k in common]})
    out_csv = slices_dir / "dataset.csv"
    df.to_csv(out_csv, index=False)
    print(f"dataset.csv saved → {out_csv}  ({human(len(df))} pairs)")


def cmd_train(
        slices_dir: Path,
        run_dir: Path,
        epochs: int,
        batch_size: int,
        img_size: int,
        workers: int,
        es_patience: int,
        es_min_delta: float,
        es_min_lr_cuts: int
):

    import torch.nn as nn
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from sklearn.model_selection import train_test_split
    import segmentation_models_pytorch as smp

    ensure_dir(run_dir)

    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("device:", device)

    df = pd.read_csv(slices_dir / "dataset.csv")
    train_df, valtest_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(valtest_df, test_size=0.5, random_state=seed, shuffle=True)
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    train_tfms = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.90, 1.10),
                 translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 rotate=(-15, 15),
                 p=0.5),
        ToTensorV2()
    ], is_check_shapes=False)
    val_tfms = A.Compose([A.Resize(img_size, img_size), ToTensorV2()], is_check_shapes=False)

    pin = (device.type == "cuda")
    pf = 4 if (workers and workers > 0) else None
    train_dl = DataLoader(
        SliceDS(train_df, train_tfms), batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=bool(workers > 0),
        prefetch_factor=pf
    )
    val_dl = DataLoader(
        SliceDS(val_df, val_tfms), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=bool(workers > 0),
        prefetch_factor=pf
    )
    test_dl = DataLoader(
        SliceDS(test_df, val_tfms), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=bool(workers > 0),
        prefetch_factor=pf
    )

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=1, classes=1).to(device)
    model = model.to(memory_format=torch.channels_last)
    print("Model device:", next(model.parameters()).device)

    bce = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary')

    def loss_fn(logits, targets):
        return 0.5 * bce(logits, targets) + 0.5 * dice_loss(logits, targets)

    @torch.no_grad()
    def iou_score(logits, targets, th=0.5, eps=1e-7):
        p = (torch.sigmoid(logits) > th).float(); t = targets.float()
        inter = (p * t).sum((1, 2, 3)); union = p.sum((1, 2, 3)) + t.sum((1, 2, 3)) - inter
        return ((inter + eps) / (union + eps)).mean().item()

    @torch.no_grad()
    def dice_score(logits, targets, th=0.5, eps=1e-7):
        p = (torch.sigmoid(logits) > th).float(); t = targets.float()
        inter = (p * t).sum((1, 2, 3)); denom = p.sum((1, 2, 3)) + t.sum((1, 2, 3))
        return ((2 * inter + eps) / (denom + eps)).mean().item()

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    try:
        scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
        def autocast_ctx(): return torch.amp.autocast(device_type=device.type, enabled=AMP)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and AMP))
        def autocast_ctx(): return torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and AMP))

    best_iou, start_epoch = 0.0, 1
    last_ckpt = run_dir / "last.pt"
    if last_ckpt.exists():
        try:
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        best_iou = ckpt.get('val_iou', 0.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}, best IoU {best_iou:.4f}")

    min_delta = float(es_min_delta)
    patience = int(es_patience)
    min_lr_cuts = int(es_min_lr_cuts)
    no_improve = 0
    lr_reductions = 0
    printed_batch_devices = False

    for epoch in range(start_epoch, epochs + 1):

        model.train(); tr_loss = 0.0; n_tr = 0
        for imgs, msks in train_dl:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            msks = msks.to(device, non_blocking=True).float()

            if not printed_batch_devices:
                print("Batch devices:", imgs.device, msks.device)
                printed_batch_devices = True

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(imgs)
                loss = loss_fn(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            tr_loss += loss.item() * imgs.size(0); n_tr += imgs.size(0)
        tr_loss /= max(1, n_tr)

        model.eval(); va_loss = va_iou = va_dice = n_va = 0.0
        with torch.no_grad():
            for imgs, msks in val_dl:
                imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                msks = msks.to(device, non_blocking=True).float()
                logits = model(imgs)
                loss = loss_fn(logits, msks)
                va_loss += loss.item() * imgs.size(0)
                va_iou  += iou_score(logits, msks) * imgs.size(0)
                va_dice += dice_score(logits, msks) * imgs.size(0)
                n_va += imgs.size(0)
        va_loss /= max(1, n_va); va_iou /= max(1, n_va); va_dice /= max(1, n_va)

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(va_iou)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            lr_reductions += 1
            print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e} | total cuts: {lr_reductions}")

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | IoU {va_iou:.4f} | Dice {va_dice:.4f}")

        payload = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_iou': va_iou, 'val_dice': va_dice
        }
        torch.save(payload, run_dir / "last.pt")
        if va_iou > best_iou + min_delta:
            best_iou = va_iou; no_improve = 0
            torch.save(payload, run_dir / "best.pt")
            print(f"New best IoU {best_iou:.4f} (saved)")
        else:
            no_improve += 1
            print(f"ES patience: {no_improve}/{patience} | LR cuts: {lr_reductions}/{min_lr_cuts}")

        if (lr_reductions >= min_lr_cuts) and (no_improve >= patience):
            print("Early stopping: plateau reached.")
            break

    print(f"Training finished. Best IoU: {best_iou:.4f}")


def cmd_eval(slices_dir: Path, run_dir: Path, img_size: int, batch_size: int, workers: int, save_grid: bool):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import segmentation_models_pytorch as smp
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    df = pd.read_csv(slices_dir / "dataset.csv")
    seed = 42
    train_df, valtest_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(valtest_df, test_size=0.5, random_state=seed, shuffle=True)

    val_tfms = A.Compose([A.Resize(img_size, img_size), ToTensorV2()], is_check_shapes=False)

    pin = torch.cuda.is_available()
    pf = 4 if (workers and workers > 0) else None
    test_dl = DataLoader(SliceDS(test_df, val_tfms), batch_size=batch_size, shuffle=False,
                         num_workers=workers, pin_memory=pin,
                         persistent_workers=bool(workers > 0),
                         prefetch_factor=pf)

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=1, classes=1).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    @torch.no_grad()
    def iou_score(logits, targets, th=0.5, eps=1e-7):
        p = (torch.sigmoid(logits) > th).float(); t = targets.float()
        inter = (p * t).sum((1, 2, 3)); union = p.sum((1, 2, 3)) + t.sum((1, 2, 3)) - inter
        return ((inter + eps) / (union + eps)).mean().item()

    @torch.no_grad()
    def dice_score(logits, targets, th=0.5, eps=1e-7):
        p = (torch.sigmoid(logits) > th).float(); t = targets.float()
        inter = (p * t).sum((1, 2, 3)); denom = p.sum((1, 2, 3)) + t.sum((1, 2, 3))
        return ((2 * inter + eps) / (denom + eps)).mean().item()

    iou_sum = dice_sum = n = 0.0
    with torch.no_grad():
        for imgs, msks in test_dl:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            msks = msks.to(device, non_blocking=True).float()
            logits = model(imgs)
            iou_sum  += iou_score(logits, msks) * imgs.size(0)
            dice_sum += dice_score(logits, msks) * imgs.size(0)
            n += imgs.size(0)
    test_iou = iou_sum / max(1, n)
    test_dice = dice_sum / max(1, n)
    (run_dir / "metrics.txt").write_text(f"test_iou={test_iou:.4f}\ntest_dice={test_dice:.4f}\n", encoding="utf-8")
    print(f"Test IoU={test_iou:.4f} | Test Dice={test_dice:.4f}  → {run_dir / 'metrics.txt'}")

    if save_grid:

        val_ds = SliceDS(val_df, val_tfms)
        k = min(6, len(val_ds))
        idxs = random.sample(range(len(val_ds)), k=k)
        import numpy as np
        plt.figure(figsize=(12, 2 * k))
        with torch.no_grad():
            for i, idx in enumerate(idxs, 1):
                img, msk = val_ds[idx]
                prob = torch.sigmoid(model(img.unsqueeze(0).to(device).to(memory_format=torch.channels_last))).cpu().squeeze().numpy()
                pred = (prob > 0.5).astype(np.uint8)
                plt.subplot(k, 3, 3 * (i - 1) + 1); plt.imshow(img.squeeze().numpy(), cmap='gray'); plt.title('Image'); plt.axis('off')
                plt.subplot(k, 3, 3 * (i - 1) + 2); plt.imshow(msk.squeeze().numpy(), cmap='gray'); plt.title('Mask');  plt.axis('off')
                plt.subplot(k, 3, 3, 3 * (i - 1) + 3) if False else None
                plt.subplot(k, 3, 3 * (i - 1) + 3); plt.imshow(pred, cmap='gray'); plt.title('Pred');  plt.axis('off')
        plt.tight_layout()
        grid_path = run_dir / "val_preds_grid.png"
        plt.savefig(grid_path, dpi=180)
        print("Saved preview grid →", grid_path)


def main():
    parser = argparse.ArgumentParser("ImageCAS local pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="Extract Kaggle ZIP")
    p_ext.add_argument("--zip", type=Path, required=True)
    p_ext.add_argument("--out", type=Path, required=True)

    p_slice = sub.add_parser("slice", help="Convert 3D volumes to 2D slices (chunked, resumable)")
    p_slice.add_argument("--in", dest="in_dir", type=Path, required=True)
    p_slice.add_argument("--out", dest="out_dir", type=Path, required=True)
    p_slice.add_argument("--chunk-size", type=int, default=50)

    p_csv = sub.add_parser("make-csv", help="Build dataset.csv from slices")
    p_csv.add_argument("--slices", type=Path, required=True)

    p_train = sub.add_parser("train", help="Train 2D U-Net on slices (resume + early stopping)")
    p_train.add_argument("--slices", type=Path, required=True)
    p_train.add_argument("--run-dir", type=Path, required=True)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--img-size", type=int, default=512)
    p_train.add_argument("--workers", type=int, default=0)

    p_train.add_argument("--es-patience", type=int, default=5,
                         help="epochs without >= min-delta improvement (after LR cuts)")
    p_train.add_argument("--es-min-delta", type=float, default=0.005, help="minimum IoU gain to count as improvement")
    p_train.add_argument("--es-min-lr-cuts", type=int, default=2,
                         help="start checking patience only after this many LR reductions")

    p_eval = sub.add_parser("eval", help="Evaluate best.pt on test set")
    p_eval.add_argument("--slices", type=Path, required=True)
    p_eval.add_argument("--run-dir", type=Path, required=True)
    p_eval.add_argument("--img-size", type=int, default=512)
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--workers", type=int, default=0)
    p_eval.add_argument("--preview-grid", action="store_true")

    args = parser.parse_args()

    if args.cmd == "extract":
        cmd_extract(args.zip, args.out)
    elif args.cmd == "slice":
        cmd_slice(args.in_dir, args.out_dir, args.chunk_size)
    elif args.cmd == "make-csv":
        cmd_make_csv(args.slices)
    elif args.cmd == "train":
        ensure_dir(args.run_dir)
        cmd_train(args.slices, args.run_dir, args.epochs, args.batch_size, args.img_size, args.workers,
                  args.es_patience, args.es_min_delta, args.es_min_lr_cuts)
    elif args.cmd == "eval":
        cmd_eval(args.slices, args.run_dir, args.img_size, args.batch_size, args.workers, args.preview_grid)


if __name__ == "__main__":

    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
