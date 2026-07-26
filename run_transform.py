import os
import pandas as pd
from multiprocessing import freeze_support
import rlwm.session as session
import rlwm.transformation as transformation  # کتابخانه اصلی ما

# مقادیر خودتان را اینجا وارد کنید
RUN_BATCH = 'my_experiments'
BASE_PATH = 'C:/Users/hosse/Documents/rlwm-main/'  # مسیر را در صورت نیاز تغییر دهید
DATA_PATH = os.path.join(BASE_PATH, 'data', RUN_BATCH)
OUTPUT_PATH = os.path.join(BASE_PATH, 'output', RUN_BATCH) # پوشه خروجی

# مطمئن شوید که پوشه خروجی وجود دارد
os.makedirs(OUTPUT_PATH, exist_ok=True)

# لیست تمام آزمودنی‌ها (مثلاً 1 و 2)
CASEIDS = list(range(1, 37))

def main():
    # ۱. بارگذاری داده‌های تمام آزمودنی‌ها
    session_list = []
    for id in CASEIDS:
        ds = session.load_session(id, DATA_PATH, suffix='')
        session_list.append(ds)
    print(f'{len(session_list)} cases loaded')

    all_trials_df_list = []

    # ۲. پردازش هر آزمودنی به صورت جداگانه
    for s in session_list:
        print(f"Processing caseid: {s.caseid}")
        
        # داده‌های خام آموزش و تست را به DataFrame تبدیل می‌کنیم
        train_df = pd.DataFrame(s.train_set, columns=['Stimulus_Pair', 'response', 'correct', 'Block'])
        test_df = pd.DataFrame(s.test_set, columns=['Stimulus_Pair', 'response', 'correct', 'Block'])
        train_df['phase'] = 'train'
        test_df['phase'] = 'test'
        
        # محاسبه ستون‌های جدید با استفاده از توابع
        pers_train, pers_test = transformation.count_trial_perseverance(s)
        delay_train, delay_test = transformation.count_trial_delay(s)
        rpred_train, rpred_test = transformation.count_trial_rpred(s)
        
        # اضافه کردن ستون‌های جدید به DataFrameها
        train_df['pers'] = pers_train
        train_df['delay'] = delay_train
        train_df['rpred'] = rpred_train
        
        # برای فاز تست، این مقادیر معمولاً 0 یا مقدار نهایی فاز آموزش هستند
        test_df['pers'] = pers_test
        test_df['delay'] = delay_test
        test_df['rpred'] = rpred_test
        
        # ادغام داده‌های آموزش و تست این آزمودنی
        subject_df = pd.concat([train_df, test_df], ignore_index=True)
        subject_df['caseid'] = s.caseid
        all_trials_df_list.append(subject_df)

    # ۳. ادغام تمام آزمودنی‌ها در یک فایل نهایی
    master_df = pd.concat(all_trials_df_list, ignore_index=True)
    
    # ۴. ذخیره فایل نهایی
    output_filename = os.path.join(OUTPUT_PATH, 'master_data_transformed.csv')
    master_df.to_csv(output_filename, index=False)
    
    print(f"Successfully created transformed data file at: {output_filename}")


if __name__ == '__main__':
    freeze_support()
    main()