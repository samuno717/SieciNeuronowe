import argparse
import time
from collections import Counter

import pandas as pd
import torch
from loguru import logger

from config import LAB5_ARTIFACT_DIR
from lab5.cnn_utils import DEVICE, set_deterministic
from lab5.transfer_utils import (
    CLASSES_PL, Strategy, build_model, get_raw_loaders, plot_resolution, plot_transfer,
    save_model_structure, train_transfer,
)
from lab5.explain_utils import Explainer, deletion_test, plot_errors_pred_vs_true, plot_explanations

EPOCHS = 6
SEED = 42
ARCHS = {"resnet18": "ResNet18", "vgg11_bn": "VGG11-BN"}

STRATEGIES = {
    "Od zera (bez pretrenowania)": Strategy("scratch", pretrained=False),
    "Zamrożony ekstraktor": Strategy("frozen", freeze_epochs=10**6),
    "Pełny FT, wspólne LR 1e-3": Strategy("full"),
    "Zamrożenie 2 ep., potem LR ekstr. 1e-4": Strategy("freeze2", freeze_epochs=2, lr_backbone=1e-4),
    "Różne LR (ekstr. 1e-4, głowa 1e-3)": Strategy("difflr", lr_backbone=1e-4),
    "Różne LR + warmup 1 ep.": Strategy("difflr_warmup", lr_backbone=1e-4, warmup_epochs=1.0),
}
RES_STRATEGIES = {
    "fine-tuning (różne LR + warmup)": "Różne LR + warmup 1 ep.",
    "zamrożony ekstraktor": "Zamrożony ekstraktor",
}
RES_SIZES = (32, 64, 128, 224)


def best_ckpt(arch: str):
    return LAB5_ARTIFACT_DIR / f"{arch}_cifar10_best.pt"


def run_training():
    loaders = get_raw_loaders(batch_size=128)
    all_summaries, all_epochs = [], []
    cache: dict[tuple, tuple[dict, dict, list[dict]]] = {}
    best_acc = {a: -1.0 for a in ARCHS}

    def run(group: str, name: str, arch: str, strat_name: str, size: int) -> dict:
        strategy = STRATEGIES[strat_name]
        key = (arch, strategy, size)
        if key not in cache:
            model, hist, summary, rows = train_transfer(group, name, arch, strategy, size, loaders, EPOCHS, seed=SEED)
            # najlepszy pretrenowany model w 224x224 zapisany dla Captum
            if strategy.pretrained and size == 224 and summary["Val Acc (koniec) [%]"] > best_acc[arch]:
                best_acc[arch] = summary["Val Acc (koniec) [%]"]
                torch.save({"state_dict": model.state_dict(), "size": size, "wariant": strat_name}, best_ckpt(arch))
                logger.info(f"Zapisano {best_ckpt(arch).name} ({strat_name}, {best_acc[arch]:.2f}%)")
            del model
            torch.cuda.empty_cache()
            cache[key] = (hist, summary, rows)
        else:
            logger.info(f"[{group}] {name}: konfiguracja policzona wcześniej - użyto zapisanego wyniku")
        hist, summary, rows = cache[key]
        all_summaries.append({**summary, "Eksperyment": group, "Wariant": name})
        all_epochs.extend({**r, "Eksperyment": group, "Wariant": name} for r in rows)
        return hist

    for arch in ARCHS:
        save_model_structure(arch)

    # 5. Strategie dotrenowania, obrazy przeskalowane do 224x224
    for arch, label in ARCHS.items():
        logger.info(f"=== EKSPERYMENT 5: strategie dotrenowania - {label}, 224x224 ===")
        group = f"5. Strategie ({label}, 224)"
        h = {name: run(group, name, arch, name, 224) for name in STRATEGIES}
        plot_transfer(h, f"{label} 224×224", f"exp5_strategie_{arch}",
                      freeze_marks={"Zamrożenie 2 ep., potem LR ekstr. 1e-4": 2})

    # 6. Przeskalowane vs nieprzeskalowane obrazy
    logger.info("=== EKSPERYMENT 6: rozdzielczość wejścia ===")
    res_rows = []
    for arch, label in ARCHS.items():
        for series, strat_name in RES_STRATEGIES.items():
            for size in RES_SIZES:
                run(f"6. Rozdzielczość ({label})", f"{label}, {series}, {size}x{size}", arch, strat_name, size)
                res_rows.append({**all_summaries[-1], "Seria": f"{label}, {series}"})
    for size in (32, 224):
        run("6. Rozdzielczość (ResNet18)", f"ResNet18, od zera, {size}x{size}", "resnet18", "Od zera (bez pretrenowania)", size)
        res_rows.append({**all_summaries[-1], "Seria": "ResNet18, od zera"})
    plot_resolution(res_rows, "exp6_rozdzielczosc")

    df_res = pd.DataFrame(res_rows)
    pivot = df_res.pivot_table(index="Seria", columns="Rozdzielczość", values="Val Acc (koniec) [%]")
    logger.info("Val Acc (koniec) w zależności od rozdzielczości:\n" + pivot.to_string())
    return pd.DataFrame(all_summaries), pd.DataFrame(all_epochs), df_res


def run_explanations():
    _, _, val_loader = get_raw_loaders(batch_size=128, n_train_eval=1)
    X_val, y_val = val_loader.X, val_loader.y
    del_rows, confusion_rows = [], []

    for arch, label in ARCHS.items():
        ckpt = torch.load(best_ckpt(arch), map_location=DEVICE)
        model = build_model(arch, pretrained=False).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        expl = Explainer(model, ckpt["size"])
        y_pred, probs = expl.predict(X_val)
        correct = (y_pred == y_val)
        logger.info(f"{label} ({ckpt['wariant']}): dokładność testowa {correct.float().mean() * 100:.2f}%")

        conf = probs.gather(1, y_pred.view(-1, 1))[:, 0]
        ok_idx = []
        for c in (0, 2, 3, 4, 6, 8):
            cand = torch.nonzero(correct & (y_val == c))[:, 0]
            if len(cand) == 0:
                continue
            sure = cand[conf[cand] > 0.9]
            ok_idx.append(int(sure[0]) if len(sure) else int(cand[conf[cand].argmax()]))
        plot_explanations(expl, X_val, y_val, y_pred, probs, ok_idx,
                          f"{label} - poprawne decyzje (mapy dla klasy przewidzianej)", f"exp7_{arch}_poprawne")

        if arch == "resnet18":
            wrong = torch.nonzero(~correct)[:, 0]
            wrong = wrong[conf[wrong].argsort(descending=True)]
            err_idx, seen = [], set()
            for i in wrong.tolist():
                if int(y_val[i]) not in seen:
                    seen.add(int(y_val[i]))
                    err_idx.append(i)
                if len(err_idx) == 6:
                    break
            plot_explanations(expl, X_val, y_val, y_pred, probs, err_idx,
                              f"{label} - błędne decyzje (mapy dla klasy przewidzianej)", f"exp7_{arch}_bledne")
            plot_errors_pred_vs_true(expl, X_val, y_val, y_pred, probs, err_idx, f"exp7_{arch}_bledne_pred_vs_true")

            pairs = Counter(zip(y_val[~correct].tolist(), y_pred[~correct].tolist()))
            n_err = int((~correct).sum())
            for (t, p), n in pairs.most_common(10):
                confusion_rows.append({"Model": label, "Klasa prawdziwa": CLASSES_PL[t], "Klasa przewidziana": CLASSES_PL[p],
                                       "Liczba": n, "Udział w błędach [%]": round(n / n_err * 100, 1)})

        g = torch.Generator().manual_seed(0)
        sample = torch.randperm(X_val.shape[0], generator=g)[:500].tolist()
        t0 = time.perf_counter()
        rows = deletion_test(expl, X_val, y_pred, sample)
        del_rows.extend({"Model": label, **r} for r in rows)
        logger.info(f"Test usuwania pikseli {label}: {time.perf_counter() - t0:.0f}s")
        del model, expl
        torch.cuda.empty_cache()

    df_del = pd.DataFrame(del_rows)
    logger.info("Test usuwania pikseli:\n" + df_del.to_string(index=False))
    return df_del, pd.DataFrame(confusion_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain-only", action="store_true", help="pomija trening, używa zapisanych modeli")
    args = parser.parse_args()

    t_total = time.perf_counter()
    set_deterministic()
    logger.info(f"Urządzenie: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    LAB5_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = LAB5_ARTIFACT_DIR / "wyniki_lab5_cz2.xlsx"

    sheets = {}
    if not args.explain_only:
        df_summary, df_epochs, df_res = run_training()
        logger.info("Podsumowanie:\n" + df_summary.to_string(index=False))
        sheets.update({"Podsumowanie": df_summary, "Historia Epok": df_epochs, "Rozdzielczość": df_res})
    elif excel_path.exists():
        old = pd.read_excel(excel_path, sheet_name=None)
        sheets.update({k: v for k, v in old.items() if k in ("Podsumowanie", "Historia Epok", "Rozdzielczość")})

    df_del, df_conf = run_explanations()
    sheets.update({"Test usuwania pikseli": df_del, "Najczęstsze pomyłki": df_conf})

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    logger.info(f"Zapisano wyniki do: {excel_path} (całkowity czas: {time.perf_counter() - t_total:.1f}s)")


if __name__ == "__main__":
    main()
