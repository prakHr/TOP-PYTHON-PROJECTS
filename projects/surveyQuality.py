import pandas as pd
import pingouin as pg

def checkSurveyQuality(survey_qs_df):
    try:
        question_sample_size = len(survey_qs_df)
        ci= 0.99
        if question_sample_size>=20:
            ci = 0.95
        cn_alpha,_ = pg.cronbach_alpha(data=survey_qs_df,ci=ci)
        quality = [
            "Excellent",
            "Good",
            "Acceptable",
            "Questionable",
            "Poor",
            "Unacceptable"
        ]
        cquality = [
            cn_alpha>=0.9,
            cn_alpha>=0.8 and cn_alpha<0.9,
            cn_alpha>=0.7 and cn_alpha<0.8,
            cn_alpha>=0.6 and cn_alpha<0.7,
            cn_alpha>=0.5 and cn_alpha<0.6,
            cn_alpha<0.5
        ]
        print(f"Detected confidence interval {ci} at question_sample_size = {question_sample_size}")
        for i,q in enumerate(cquality):
            if q == True:
                return quality[i]
        return -1
    except Exception as e:
        return str(e)

survey_qs_df = pd.DataFrame({
    'Q1': [1,2,3,2],
    'Q2': [1,2,3,2],
    'Q3': [1,2,3,2]
})
rv = checkSurveyQuality(survey_qs_df)
print(rv)