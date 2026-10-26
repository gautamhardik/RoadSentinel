import csv

with open('experiments/phase12_p2_architecture/checkpoints/results.csv') as f:
    print(f"{'Epoch':>6} | {'Precision':>10} | {'Recall':>10} | {'mAP50':>10} | {'mAP50-95':>10} | {'Train Box':>10} | {'Val Box':>10}")
    print("-" * 80)
    for row in csv.DictReader(f):
        ep = row['epoch']
        prec = float(row['metrics/precision(B)'])
        rec = float(row['metrics/recall(B)'])
        map50 = float(row['metrics/mAP50(B)'])
        map95 = float(row['metrics/mAP50-95(B)'])
        tbox = float(row['train/box_loss']) if row['train/box_loss'] != 'nan' else 0.0
        vbox = float(row['val/box_loss']) if row['val/box_loss'] != 'nan' else 0.0
        if map50 > 0:
            print(f"{ep:>6} | {prec*100:>9.2f}% | {rec*100:>9.2f}% | {map50*100:>9.2f}% | {map95*100:>9.2f}% | {tbox:>10.4f} | {vbox:>10.4f}")
