# Рисунки для розділу 4 (етап 8)

Згенеровано `ml/figures_stage8.py`; PNG 300 dpi і PDF, ширина ≈ 16 см, без заголовків (підпис — у тексті).

| Файл | Що показує | Джерело |
|---|---|---|
| `fig1_apfd_test` | APFD за методами на test, точка + 95 % bootstrap-інтервал за 61 «червоним» комітом | `main/per_commit_test.csv` (+ `apfd_bootstrap.csv`) |
| `fig2_recall_first_vs_time` | криві recall перших появ від фактичного часу (розвідувально, не перевірка H1) | `cost_curve/mean_curves.csv` |
| `fig3_rolling_transfer` | перенесення τ (RF) і B (комбінована) між val і test вікон rolling (expanding) | `rolling/per_fold.csv` |
| `fig4_ablation` | ablation: APFD і recall перших появ / час за τ за наборами ознак | `ablation/summary.csv` |
| `fig5_closed_loop` | замкнена петля: частка невиявлених до виправлення дефектів і затримка залежно від N | `closed_loop/summary.csv` |
| `fig6_learning_curve` | learning curve: APFD і recall / час за τ залежно від кількості навчальних комітів | `learning_curve/summary.csv` |
| `fig7_error_categories` | категорії пропущених перших появ (а)–(д) і контраст зі спійманими | `errors/categories.csv` |
