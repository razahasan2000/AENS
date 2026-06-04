"""Verify paper numbers match JSON files."""
import json, os, re

OUT = 'AE_NS_Results_v2'

# 1. Check Table 1 v2 AE-NS numbers
print('=== Table 1: AE-NS v2 standalone ===')
for ds in ['adult', 'compas', 'credit']:
    with open(os.path.join(OUT, f'v2_{ds}_results.json')) as f:
        v2 = json.load(f)
    a = v2['aggregate']
    print(f'  {ds}: acc={a["accuracy"]["mean"]:.4f}+/-{a["accuracy"]["std"]:.4f} dpd={a["dpd"]["mean"]:.4f}+/-{a["dpd"]["std"]:.4f}')

# 2. Check Fairpost numbers
print('\n=== Table 1: AE-NS Z + Fairlearn ===')
for ds in ['adult', 'compas', 'credit']:
    with open(os.path.join(OUT, f'fairpost_3seeds_{ds}.json')) as f:
        fp = json.load(f)
    b = fp['aggregate']
    print(f'  {ds}: acc={b["accuracy"]["mean"]:.4f}+/-{b["accuracy"]["std"]:.4f} dpd={b["dpd"]["mean"]:.4f}+/-{b["dpd"]["std"]:.4f}')

# 3. Check NMI + classify-on-Z
print('\n=== Tables 2-3: NMI + classify-on-Z ===')
for ds in ['adult', 'compas', 'credit']:
    with open(os.path.join(OUT, f'{ds}_revision.json')) as f:
        rev = json.load(f)
    cz = rev['classify_on_Z']
    print(f'  {ds}: NMI={rev["nmi"]:.4f} acc_Z={cz["acc_on_Z"]:.4f} acc_X={cz["acc_on_X"]:.4f} dpd_Z={cz["dpd_on_Z"]:.4f} dpd_X={cz["dpd_on_X"]:.4f}')

# 4. Check figures exist
print('\n=== Figures ===')
figs = os.listdir('figures')
pdfs = [f for f in figs if f.endswith('.pdf')]
pngs = [f for f in figs if f.endswith('.png')]
print(f'  {len(pdfs)} PDF + {len(pngs)} PNG files')
for f in sorted(pdfs):
    print(f'    {f}')

# 5. Check paper has correct Table 1 numbers
print('\n=== Cross-check paper Table 1 ===')
with open('paper_revision.md', 'r', encoding='utf-8') as f:
    paper = f.read()

# Check for key v2 numbers in paper
checks = [
    ('Adult AE-NS v2 acc', '0.781', 'adult v2 acc'),
    ('Adult AE-NS v2 dpd', '0.025', 'adult v2 dpd'),
    ('COMPAS AE-NS v2 acc', '0.640', 'compas v2 acc'),
    ('COMPAS AE-NS v2 dpd', '0.145', 'compas v2 dpd'),
    ('Credit AE-NS v2 acc', '0.797', 'credit v2 acc'),
    ('Credit AE-NS v2 dpd', '0.025', 'credit v2 dpd'),
    ('Adult Fairpost acc', '0.824', 'adult fairpost acc'),
    ('Adult Fairpost dpd', '0.009', 'adult fairpost dpd'),
    ('COMPAS Fairpost acc', '0.653', 'compas fairpost acc'),
    ('COMPAS Fairpost dpd', '0.035', 'compas fairpost dpd'),
    ('Credit Fairpost acc', '0.820', 'credit fairpost acc'),
    ('Credit Fairpost dpd', '0.009', 'credit fairpost dpd'),
]
for label, num, key in checks:
    found = num in paper
    status = 'OK' if found else 'MISSING'
    print(f'  {status}: {label} = {num}')

print('\nDone!')
