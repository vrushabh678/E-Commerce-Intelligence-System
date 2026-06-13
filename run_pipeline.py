import os, subprocess, time

phases = [
    ('Phase 1: ETL & Schema', 'main.py'),
    ('Phase 2: Root Cause Analysis', 'phase2_rca.py'),
    ('Phase 3: Cohorts & LTV', 'phase3_cohorts.py'),
    ('Phase 4: RFM Segmentation', 'phase4_rfm.py'),
    ('Phase 5: Funnel Analysis', 'phase5_funnel.py'),
    ('Phase 6: ML Forecasting', 'phase6_forecast.py'),
    ('Phase 7: Dashboard & Delivery', 'Phase7_Delivery.py')
]

print('🚀 Starting E-Commerce Analytics Pipeline...')
start = time.time()

for name, script in phases:
    print(f'\n➤ Running {name}...')
    result = subprocess.run(['python', script])
    if result.returncode != 0:
        print(f'❌ Error in {script}. Pipeline halted.')
        break
    print(f'✅ {name} Completed.')

print(f'\n🎉 Pipeline finished in {time.time() - start:.2f} seconds!')