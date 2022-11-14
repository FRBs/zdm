from zdm.craco import loading
import time
import numpy as np
import os
from pkg_resources import resource_filename

i = 30
res_list = []

for i in range(i):
    st = time.process_time()

    isurvey, igrid = loading.survey_and_grid(
        survey_name="CRACO_std_May2022",
        NFRB=100,
        lum_func=3,
        # sdir=os.path.join(resource_filename("zdm", "data"), "Surveys/James2022a"),
    )

    et = time.process_time()
    res = et - st
    res_list.append(res)

avg_res = np.mean(res_list)

print("CPU Execution time average:", avg_res, "seconds")
