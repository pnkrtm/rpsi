SET model=4_2
SET input_model=../work_models/model_%model%
title %input_model%
python run_inversion.py -i %input_model% -nx 80 -dx 20
pause