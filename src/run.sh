#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --account=m5264

conda activate py313   # analysis environment (see requirements.txt)

python run_heatwave_ensemble.py 2011_CTRL_p0 2011_CTRL_p12 2011_CTRL_n12 2011_CTRL_p24 2011_CTRL_n24 2011_SSP245_p0 2011_SSP245_p12 2011_SSP245_n12 2011_SSP245_p24 2011_SSP245_n24
python run_heatwave_ensemble.py ERA5_obs_2021 2021_CTRL_p0 2021_CTRL_p24 2021_CTRL_n24 2021_SSP245_p0 2021_SSP245_p24 2021_SSP245_n24

for yr in 2011 2021; do
  for th in 0.2 0.4 0.6 0.8; do
    python analyze_tell_heatwave.py --year $yr --threshold $th
  done
  python plot_affected_ba.py --year $yr          # needs all four thresholds
  python interconnection_stress.py --year $yr    # needs tell_load_daily_all_<year>.csv
done


