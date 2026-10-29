# BAM-CCM v0.6

Исходная реализация **Bayesian Adaptive Metric Convergent Cross Mapping**.

Метод восстанавливает один временной ряд по текущим и прошлым значениям другого. В отличие от классического CCM, он обучает расстояния между состояниями с учётом важности координат и использует регрессию на основе гауссовского процесса (GP).

Реализованы два подхода:
- **BAM-MAP** — обучение параметров с регуляризацией.
- **BAM-Bayes** — байесовское усреднение прогнозов с использованием MCMC.

В предварительных экспериментах BAM-CCM v0.6 показал меньшую ошибку восстановления (RMSE), чем классический CCM, в 78 из 90 сравнений (86,7%). При этом более простая модель GP-ARD в отдельном сравнении показала меньшую среднюю RMSE.

Подробные результаты, математическое описание и сравнение с v0.7 представлены в [главном README](../README.md) и [полном научном отчёте](../report/main.pdf).

## Установка и запуск

Все команды выполняются из корневой папки репозитория. Требуется Python 3.10+

**1. Установка зависимостей (macOS / Linux)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r original/requirements.txt
```

**2. Проверка программы**

```bash
python original/bam_ccm.py --self-test
```

**3. Пробный запуск на синтетических данных**

```bash
python original/bam_ccm.py --dataset synthetic --system logistic \
  --max-train 50 --map-maxiter 5 --mcmc-warmup 10 \
  --mcmc-steps 12 --no-curves
```

**4. Сравнение на независимых траекториях**

```bash
python original/bam_ccm.py --benchmark --repeats 5 \
  --noise 0.10 --coupling 0.05 --max-train 65 \
  --map-maxiter 35 --mcmc-warmup 60 --mcmc-steps 100
```

Полный список параметров:

```bash
python original/bam_ccm.py --help
```

## Данные и результаты

- `bam_ccm.py` — основная реализация алгоритма
- `data/` — синтетические и реальные данные.
- `tests/` — тесты
- `results/` — результаты запусков, таблицы и графики


