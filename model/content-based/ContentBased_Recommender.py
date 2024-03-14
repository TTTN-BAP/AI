import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sortedcontainers import SortedList
from datetime import datetime

class ContentBased:
    def __init__(self):
        df = pd.read_csv('./dataset/job_data_final.csv')

        df['id'] = range(1, len(df) + 1)
        df['job_level_text'] = df['job_level']
        unique_job_levels = df['job_level'].unique()
        job_level_mapping = {
            'Nhân viên': 1,
            'Mới tốt nghiệp': 2,
            'Quản lý': 3,
            'Trưởng nhóm / Giám sát': 4,
            'Giám đốc': 5,
            'Tổng giám đốc': 6,
            'Phó Giám đốc': 7,
            'Sinh viên/ Thực tập sinh': 8
        }
        df['job_level'] = df['job_level'].map(job_level_mapping).fillna(0).astype(int)
        
        df = df.dropna(thresh=df.shape[1]-3)
        df['job_name'] =df['job_name'].fillna('')
        
        df['job_experience_text'] = df['job_experience']
        def get_average_experience(string):
            numbers = [int(s) for s in string.split() if s.isdigit()]
            if numbers:
                return sum(numbers) / len(numbers)
            else:
                return 0
        df['job_experience'] = df['job_experience'].apply(get_average_experience)

        df['job_salary_text'] = df['job_salary']
        def process_salary_range(salary_str):
            if isinstance(salary_str, str):
                numbers = [int(s) for s in salary_str.split() if s.isdigit()]
                
                if 'cạnh tranh' in salary_str.lower():
                    return np.nan
                elif numbers:
                    return sum(numbers) / len(numbers)
                else:
                    return None
            else:
                return None
        df['job_salary'] = df['job_salary'].apply(process_salary_range)
        
        self.jobs = {}
        for row in df.values:
            link, job_name,id_company,company_name,job_salary,job_experience,job_level,job_expired_date,job_details,job_required, company_logo, job_address, id, job_level_text, job_experience_text, job_salary_text =row
            self.jobs[id] = {
                'id': id,
                'job_name': job_name,
                'id_company': id_company,
                'company_name': company_name,
                'job_salary': job_salary,
                'job_salary_text': job_salary_text,
                'job_experience': job_experience,
                'job_experience_text': job_experience_text,
                'job_level': job_level,
                'job_level_text': job_level_text,
                'job_expired_date':job_expired_date,
                'job_details': job_details,
                'job_required':job_required,
                'company_logo':company_logo,
                'job_address':job_address
            }

    def get_sim_job_cv_target(self, cv_target, job_id_i):
        job_target = set(self.jobs[job_id_i]['job_name'])
        cv_target = set(cv_target)
        return  0.3 * len(cv_target.intersection(job_target)) / len(cv_target.union(job_target))

    def get_sim_job_cv_academi_level(self, cv_academi_level, job_id_i):
        job_required = set(self.jobs[job_id_i]['job_required'])
        cv_academi_level = set(cv_academi_level)
        return  0.2 * len(cv_academi_level.intersection(job_required)) / len(cv_academi_level.union(job_required))
    
    def get_sim_job_cv_skill(self, cv_skill, job_id_i):
        job_required = set(self.jobs[job_id_i]['job_required'])
        cv_skill = set(cv_skill)
        return  0.3 * len(cv_skill.intersection(job_required)) / len(cv_skill.union(job_required))

    def get_sim_job_cv_interested(self, cv_interested, job_id_i):
        job_details = set(self.jobs[job_id_i]['job_details'])
        cv_interested = set(cv_interested)
        return  0.2 * len(cv_interested.intersection(job_details)) / len(cv_interested.union(job_details))

    def get_job_cv_similarities(self, cv_target, cv_academi_level, cv_skill, cv_interested, job_id_i):
        sim_academi_level = self.get_sim_job_cv_academi_level(cv_academi_level, job_id_i)
        sim_target = self.get_sim_job_cv_target(cv_target, job_id_i)
        sim_required = self.get_sim_job_cv_skill(cv_skill, job_id_i)
        sim_details = self.get_sim_job_cv_interested(cv_interested, job_id_i)
    
        return sim_target + sim_required + sim_details + sim_academi_level
    
    def recommend_with_cv(self, cv_target, cv_academi_level, cv_skill, cv_interested):
        k = 10
        list = SortedList()
        for id in self.jobs.keys():
            #print(id)
            sim = self.get_job_cv_similarities(cv_target,cv_academi_level, cv_skill, cv_interested, id)
            list.add((sim, id))
            if len(list) > k:
                del list[0]
        sorted_list = sorted(list, reverse=True) 
        result = []
        for score, id in sorted_list:
            result.append({
                'score': round(score, 2),
                'id': id,
                'job_name': self.jobs[id]['job_name'],
                'id_company': self.jobs[id]['id_company'],
                'company_name': self.jobs[id]['company_name'],
                #'job_salary_text': self.jobs[id]['job_salary_text'],
                'job_salary': self.jobs[id]['job_salary_text'],
                #'job_experience': self.jobs[id]['job_experience'],
                'job_experience': self.jobs[id]['job_experience_text'],
                #'job_level': self.jobs[id]['job_level'],
                'job_level': self.jobs[id]['job_level_text'],
                'job_expired_date': self.jobs[id]['job_expired_date'],
                'job_details': self.jobs[id]['job_details'],
                'job_required': self.jobs[id]['job_required'],
                'company_logo': self.jobs[id]['company_logo'],
                'job_address': self.jobs[id]['job_address']
            })
        return result
    
    def get_sim_job_name(self, job_id_i, job_id_j):
        job_name_i = set(self.jobs[job_id_i]['job_name'])
        job_name_j = set(self.jobs[job_id_j]['job_name'])

        return  0.4 * len(job_name_i.intersection(job_name_j)) / len(job_name_i.union(job_name_j))

    def get_sim_id_company(self, job_id_i, job_id_j):
        id_company_i = self.jobs[job_id_i]['id_company']
        id_company_j = self.jobs[job_id_j]['id_company']

        if  id_company_i == id_company_j:
            return 0.1
        return  0

    def get_sim_job_experience(self, job_id_i,job_id_j):
        job_experience_i = float(self.jobs[job_id_i]['job_experience'])
        job_experience_j = float(self.jobs[job_id_j]['job_experience'])

        if job_experience_i == 0 or job_experience_j == 0:
            return 0

        diff = abs(job_experience_i - job_experience_j)
        if diff < 0.25:
            return 0.2
        elif diff < 0.5:
            return 0.2 * 0.8
        elif diff < 0.75:
            return 0.2 * 0.6
        elif diff < 1:
            return 0.2 * 0.4
        return 0

    def get_sim_job_salary(self, job_id_i,job_id_j):
        job_salary_i = float(self.jobs[job_id_i]['job_salary'])
        job_salary_j = float(self.jobs[job_id_j]['job_salary'])

        diff = abs(job_salary_i - job_salary_j)
        if diff < 1:
            return 0.2
        elif diff < 5:
            return 0.2 * 0.8
        elif diff < 8:
            return 0.2 * 0.6
        elif diff < 10:
            return 0.2 * 0.4
        return 0
        

    def get_sim_job_level(self, job_id_i,job_id_j):
        job_level_i = int(self.jobs[job_id_i]['job_level'])
        job_level_j = int(self.jobs[job_id_j]['job_level'])
        if job_level_i == job_level_j: 
            return 0.1
        return  0

    def get_job_similarities(self,job_id_i, job_id_j):
        sim_name = self.get_sim_job_name(job_id_i, job_id_j)
        sim_id_company = self.get_sim_id_company(job_id_i, job_id_j)
        sim_experience = self.get_sim_job_experience(job_id_i, job_id_j)
        sim_salary = self.get_sim_job_salary(job_id_i, job_id_j)
        sim_level = self.get_sim_job_level(job_id_i, job_id_j)
        return sim_name + sim_id_company + sim_experience + sim_salary + sim_level 
    
    def recommend_with_job(self, job_id):
        k = 10
        list = SortedList()
        for id in self.jobs.keys():
            if (id == job_id):
                continue
            sim = self.get_job_similarities(job_id, id)
            list.add((sim, id))
            if len(list) > k:
                del list[0]
        sorted_list = sorted(list, reverse=True) 
        result = []
        for score, id in sorted_list:
            result.append({
                'score': round(score, 2),
                'id': id,
                'job_name': self.jobs[id]['job_name'],
                'id_company': self.jobs[id]['id_company'],
                'company_name': self.jobs[id]['company_name'],
                #'job_salary_text': self.jobs[id]['job_salary_text'],
                'job_salary': self.jobs[id]['job_salary_text'],
                #'job_experience': self.jobs[id]['job_experience'],
                'job_experience': self.jobs[id]['job_experience_text'],
                #'job_level': self.jobs[id]['job_level'],
                'job_level': self.jobs[id]['job_level_text'],
                'job_expired_date': self.jobs[id]['job_expired_date'],
                'job_details': self.jobs[id]['job_details'],
                'job_required': self.jobs[id]['job_required'],
                'company_logo': self.jobs[id]['company_logo'],
                'job_address': self.jobs[id]['job_address']
            })
        return result