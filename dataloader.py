"""
Author: Srinidhi Havaldar
Date: January 08, 2019
"""

import json
import numpy as np
import pandas as pd


class DataLoader:
    """
    A generic class to load data from a configuration file file

    Args:
        conf_file (str): path to the configuration file

    Returns:
        data (list)
    """

    def __init__(self, conf_file):
        self.conf_file = conf_file

        with open(self.conf_file, 'r') as f:
            config = json.load(f)

        self.data_file = config["COLUMNS"]["DATA_FILEPATH"]
        self.conn = config["COLUMNS"]["CONNECTION_TYPE"]
        self.lesson_type = config["COLUMNS"]["LESSONTYPE_COL"]
        self.student = config["COLUMNS"]["STUDENT_COL"]
        self.mlo_id = config["COLUMNS"]["MLOID_COL"]
        self.qcode = config["COLUMNS"]["QUESTION_CODE_COL"]
        self.start_time = config["COLUMNS"]["STARTTIME_COL"]
        self.q_score = config["COLUMNS"]["SCORE_COL"]
        self.subject_name = config["COLUMNS"]["SUBJECT_NAME_COL"]
        self.guess_bnd = config["COLUMNS"]["GUESS_BOUND"]
        self.slip_bnd = config["COLUMNS"]["SLIP_BOUND"]
        self.solver = config["COLUMNS"]["OPT_METHOD"]
        self.tolerance = config["COLUMNS"]["TOLERANCE"]
        self.m_iters = config["COLUMNS"]["MAX_ITERS"]
        self.param_f = config["COLUMNS"]["APPLIED_MODEL_PARAMS"]

    def read_data(self):
        if self.conn == 'csv':
            dt = pd.read_csv(self.data_file)
            dt[self.lesson_type] = dt[self.lesson_type].str.strip()
            dt[self.lesson_type] = dt[self.lesson_type].str.lower()
            
            # mask_subject = np.in1d(dt[self.subject_name].values, 'Math')
            # dt = dt[mask_subject]
            dt[self.subject_name] = dt[self.subject_name].str.strip()
            dt[self.subject_name] = dt[self.subject_name].str.lower()
            mask_teq = np.in1d(dt[self.lesson_type].values, ['teq_1', 'teq_2', 'teq_3', 'sa'])
            dt = dt[mask_teq]
            
            dt[self.q_score] = np.where(dt[self.q_score] >= 0.5, 1, 0)
            # df = dt[[self.student, self.start_time, self.endtime, self.mlo_id, self.lo_title, self.q_score]]
            df = dt.sort_values(by=[self.start_time, self.mlo_id, self.qcode, self.student],
                                ascending=[True, True, True, True])
            
            df.reset_index(drop=True, inplace=True)
            data = df.T.reset_index().values.T.tolist()
            
            return data
