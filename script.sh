config_path='/PATH_TO/MUS-CDB/configs/redet/redet_re50_refpn_1x_dota_le90.py'
sample="mus-cdb"
c_0_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle0/epoch_12.pth"
s_0_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle0/example_Task1"
c_1_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle1/epoch_12.pth"
s_1_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle1/example_Task1"
c_2_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle2/epoch_12.pth"
s_2_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle2/example_Task1"
c_3_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle3/epoch_12.pth"
s_3_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle3/example_Task1"
c_4_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle4/epoch_12.pth"
s_4_path="/PATH_TO/MUS-CDB/work_dirs/PU/EXP0/cycle4/example_Task1"
work_dir="/PATH_TO/MUS-CDB/work_dirs/PU"
python /PATH_TO/MUS-CDB/train.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 0 &&
python /PATH_TO/MUS-CDB/test.py --config $config_path --checkpoint $c_0_path --submission-dir $s_0_path &&
python /PATH_TO/MUS-CDB/train.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 1 &&
python /PATH_TO/MUS-CDB/test.py --config $config_path --checkpoint $c_1_path --submission-dir $s_1_path &&
python /PATH_TO/MUS-CDB/train.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 2 &&
python /PATH_TO/MUS-CDB/test.py --config $config_path --checkpoint $c_2_path --submission-dir $s_2_path &&
python /PATH_TO/MUS-CDB/train.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 3 &&
python /PATH_TO/MUS-CDB/test.py --config $config_path --checkpoint $c_3_path --submission-dir $s_3_path &&
python /PATH_TO/MUS-CDB/train.py --config $config_path --work-dir $work_dir --al-sample $sample --cycle 4 &&
python /PATH_TO/MUS-CDB/test.py --config $config_path --checkpoint $c_4_path --submission-dir $s_4_path