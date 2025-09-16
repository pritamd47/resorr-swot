import pandas as pd
import numpy as np
import os

def filtering(res_sa_folder, res_list, res_nomArea, filt_1_thresh=1, filt_2_thresh=5, filt_3_thresh=.1, satellites=['l8', 'l9', 's2', 's1']):
    """TMS-OS with aggressive filtering (Suresh et al., 2023). Filter and correct satellite based surface area time-series using Sentinel-1 (SAR) data.

    Args:
        res_sa_folder (_type_): path of directory containing surface area values. It is the `gee_sarea_tmsos` folder in the `data` directory of the RAT project.
        res_list (_type_): a list of reservoir names - must be same as the names of the files in the `res_sa_folder` directory. These reservoirs will be processed.
        res_nomArea (_type_): list of nominal reservoir surface areas in km^2. The values should correspond to the reservoirs in `res_list` in the same order.
        filt_1_thresh (int, optional): Filtering threshold 1: Sigma from monthly mean (optical data only). Defaults to 1.
        filt_2_thresh (int, optional): Filtering threshold 2: Percentage of the nominal surface area deviation from Sentinel-1 estimates (in percentage). Defaults to 5.
        filt_3_thresh (float, optional): Filtering threshold 3: Monthly sigma deviation from area change trend estimated using Sentinel-1 (avg. of previous 2 SAR trends). Defaults to .1.
        satellites (list, optional): List of satellites to use. Valid values are 'l9' for Landsat-9, 'l8' for Landsat-8, 's2' for Sentinel-2, 's1' or 'sar' for Sentinel-1. Defaults to ['l8', 'l9', 's2', 's1'].
    """
    # check if the combination of satellites will work
    assert 's1' in satellites or 'sar' in satellites, 'SAR data is required'
    assert 'l8' in satellites or 'l9' in satellites or 's2' in satellites, 'Optical data is required'

    data = []

    for curr_dam_index,res_name in enumerate(res_list):   
        optical_to_concat = []
        if 'l8' in satellites:
            res_wa_timeseries_l8 =  pd.read_csv(f"{res_sa_folder}/l8/{res_name}.csv")
            res_wa_timeseries_l8['area'] = res_wa_timeseries_l8['corrected_area_cordeiro']
            res_wa_timeseries_l8 = res_wa_timeseries_l8.fillna(0)
            res_wa_timeseries_l8['time'] = pd.to_datetime(res_wa_timeseries_l8['from_date'])
            res_wa_timeseries_l8['date'] = res_wa_timeseries_l8['time'].dt.date
            optical_to_concat.append(res_wa_timeseries_l8)
        if 's2' in satellites:
            res_wa_timeseries_s2 =  pd.read_csv(f"{res_sa_folder}/s2/{res_name}.csv")
            res_wa_timeseries_s2['area'] = res_wa_timeseries_s2['water_area_corrected']
            res_wa_timeseries_s2 = res_wa_timeseries_s2.fillna(0)
            res_wa_timeseries_s2['time'] = pd.to_datetime(res_wa_timeseries_s2['date'])
            res_wa_timeseries_s2['date'] = res_wa_timeseries_s2['time'].dt.date
            optical_to_concat.append(res_wa_timeseries_s2)
        if 's1' in satellites:
            res_wa_timeseries_s1 =  pd.read_csv(f"{res_sa_folder}/sar/{res_name}_12d_sar.csv")
            # pre-processing mspc data - dates are converted to datetime objects and set as the index. Sentinel-2 and Landsat-8 datasets are merged and the dataset sorted w.r.t date.
            res_wa_timeseries_s1['time'] = pd.to_datetime(res_wa_timeseries_s1['time'])
            res_wa_timeseries_s1['date'] = res_wa_timeseries_s1['time'].dt.date

        #Merging dataframes
        res_wa_1to5day = pd.concat(optical_to_concat)
        res_wa_1to5day = res_wa_1to5day.sort_values('date')
        res_wa_1to5day['WA_mean'] = res_wa_1to5day['area']

        #Setting date as index and creating copies for further manipulation
        pdf1 = res_wa_1to5day.copy()
        pdf2 = res_wa_timeseries_s1.copy()
        pdf1 = pdf1.set_index('date')
        pdf2 = pdf2.set_index('date')

        #Dropping duplicates and reindexing the Sentinel-2 dataset to match the frequency of the 1-5day Optical dataset.
        pdf1['date_integer'] = pdf1.index
        pdf1['date_integer'] = pd.DatetimeIndex(pdf1.index).strftime('%Y%m%d').astype(int)
        pdf2['date_integer'] = pdf2.index
        pdf2['date_integer'] = pd.DatetimeIndex(pdf2.index).strftime('%Y%m%d').astype(int)
        pdf1 = pdf1.drop_duplicates(subset = ['date_integer'], keep='first')
        pdf2 = pdf2.drop_duplicates(subset = ['date_integer'], keep='first')
        pdf2 = pdf2.reindex(pdf1.index)
        pdf2['water_area'] = pdf2['sarea'].interpolate()

        pdf2.drop('time', axis = 1, inplace = True)
        pdf2['date_integer'] = pd.DatetimeIndex(pdf2.index).strftime('%Y%m%d').astype(int)

        #pre-processing for Filter-1.
        merged_pdf = pdf2.copy()
        merged_pdf = merged_pdf.rename(columns = {'water_area':'WA_SAR'})
        merged_pdf['WA_Optical'] = pdf1['WA_mean']
        merged_pdf.drop(merged_pdf.index[0:3], inplace = True)

        ## Filtering - Step 1 -- Removing outliers by considering monthly mean
        merged_pdf_2 = merged_pdf.copy()
        merged_pdf_2['Date'] = pd.to_datetime(merged_pdf_2.index)
        merged_pdf_2['YearMonth'] = merged_pdf_2['Date'].dt.year.astype(str) + merged_pdf_2['Date'].dt.month.astype(str)
        grouped_pdf = merged_pdf_2.groupby('YearMonth')

        monthly_mean_Optical = grouped_pdf['WA_Optical'].mean()
        monthly_std_Optical  = grouped_pdf['WA_Optical'].std()

        merged_pdf_2 = pd.merge(merged_pdf_2, monthly_mean_Optical, on='YearMonth', suffixes=('', '_monthly_mean'))
        merged_pdf_2 = pd.merge(merged_pdf_2, monthly_std_Optical, on='YearMonth', suffixes=('', '_monthly_std'))

        merged_pdf_2['WA_Optical_previous'] = merged_pdf_2['WA_Optical'].shift(1)  # Create a new column with the previous value of 'WA_Optical'

        for index, row in merged_pdf_2.iterrows():
                monthly_mean_Optical = row['WA_Optical_monthly_mean']
                monthly_std_Optical = row['WA_Optical_monthly_std']               
                
                if (row['WA_Optical'] < (monthly_mean_Optical - filt_1_thresh*monthly_std_Optical) or row['WA_Optical'] > (monthly_mean_Optical + filt_1_thresh*monthly_std_Optical)):
                    # print(row['WA_Optical'], monthly_mean, monthly_std)
                    merged_pdf_2.loc[index, 'WA_Optical'] = row['WA_Optical_previous']            
                
        merged_pdf_2.set_index('Date', inplace = True)
        merged_pdf_remOutliers = merged_pdf_2.copy()
        merged_pdf_remOutliers['deviations'] = merged_pdf_remOutliers['WA_Optical'] - merged_pdf_remOutliers['WA_SAR']

        dev_bias = merged_pdf_remOutliers['deviations'].median()

        merged_pdf_remOutliers['norm_deviations'] = dev_bias = merged_pdf_remOutliers['deviations'] - dev_bias

        #Filtering 2 - SAR bias correction
        res_nom_SA = res_nomArea[curr_dam_index] #km^2
        cloud_thresh = -1 #%
        filt2_thresh_values = (-res_nom_SA*filt_2_thresh/100, res_nom_SA*filt_2_thresh/100)

        merged_pdf_filt2 = merged_pdf_remOutliers.copy()
        # startPoint = 10
        # merged_pdf_filt2 = merged_pdf_filt2[startPoint:]

        outliers = ((merged_pdf_filt2['deviations'] < filt2_thresh_values[0]) | (merged_pdf_filt2['deviations'] > filt2_thresh_values[1]))

        for i, val in merged_pdf_filt2.loc[outliers, 'deviations'].items():  
            # mod_val = merged_pdf_filt2['WA_Optical'].loc[:i][-1] - val - res_nom_SA*filt2_thresh/100
            
            opt_val = merged_pdf_filt2['WA_Optical'].loc[:i][-1]    
            sar_val = merged_pdf_filt2['WA_SAR'].loc[:i][-1]
            opt_sar_dev = merged_pdf_filt2['deviations'].loc[:i][-1]

            dev_sign = np.abs(merged_pdf_filt2['deviations'].loc[:i][-1]/merged_pdf_filt2['deviations'].loc[:i][-1])
            mod_val = opt_val + dev_sign* res_nom_SA*filt_2_thresh/100 - opt_sar_dev
            
            merged_pdf_filt2.loc[i, 'WA_Optical'] = mod_val

        #Filtering 3 - SAR Trend correction
        merged_pdf_filt3 = merged_pdf_filt2.copy()
        merged_pdf_filt3['date_integer'] = merged_pdf_filt3.index
        merged_pdf_filt3['date_integer'] = pd.DatetimeIndex(merged_pdf_filt3.index).strftime('%Y%m%d').astype(int)
        merged_pdf_filt3['timeDiff'] = merged_pdf_filt3['date_integer'].diff()
        merged_pdf_filt3['SAR_diff'] = merged_pdf_filt3['WA_SAR'].diff()
        merged_pdf_filt3['SAR_diff'].fillna(0, inplace = True)
        merged_pdf_filt3 = merged_pdf_filt3.drop_duplicates(subset=['date_integer'], keep='first')
        merged_pdf_filt3['WA_Optical_cor'] = merged_pdf_filt3['WA_Optical']

        merged_pdf_filt3['SAR_trend'] = merged_pdf_filt3['WA_SAR'].pct_change()
        merged_pdf_filt3['WA_SAR_shift'] = merged_pdf_filt3['WA_SAR'].shift(1)
        merged_pdf_filt3['WA_Optical_cor_trend'] = merged_pdf_filt3['WA_Optical_cor'].pct_change()

        merged_pdf_filt3['SAR_trend_dev_Optical'] = np.abs(merged_pdf_filt3['WA_Optical_cor_trend'] - merged_pdf_filt3['SAR_trend'])
        grouped_pdf_2 = merged_pdf_filt3.groupby('YearMonth')
        monthly_mean_sar_trend_dev_optical = grouped_pdf_2['SAR_trend_dev_Optical'].mean()
        monthly_std_sar_trend_dev_optical = grouped_pdf_2['SAR_trend_dev_Optical'].std()

        merged_pdf_filt3['date'] = pd.to_datetime(merged_pdf_filt3.index)
        merged_pdf_filt3 = pd.merge(merged_pdf_filt3, monthly_mean_sar_trend_dev_optical, on='YearMonth', suffixes=('', '_monthly_mean'))
        merged_pdf_filt3 = pd.merge(merged_pdf_filt3, monthly_std_sar_trend_dev_optical, on='YearMonth', suffixes=('', '_monthly_std'))
        merged_pdf_filt3.set_index('date', inplace = True)

        for i, val in merged_pdf_filt3['SAR_trend'][3:].items():
            sar_trend_weekly_av = (merged_pdf_filt3['SAR_trend'].loc[:i][-2] + merged_pdf_filt3['SAR_trend'].loc[:i][-3])/2
            curr_sar_trend = merged_pdf_filt3['SAR_trend'].loc[:i][-1]
            
            curr_opt_trend = merged_pdf_filt3['WA_Optical_cor_trend'].loc[:i][-1]
            curr_opt_trend_2 = (merged_pdf_filt3['WA_Optical_cor_trend'].loc[:i][-1] + merged_pdf_filt3['WA_Optical_cor_trend'].loc[:i][-2])/2
            
            # Checking if optical trend is significantly greater than sar trend 
            if np.abs(curr_opt_trend_2 - curr_sar_trend) > filt_3_thresh* merged_pdf_filt3['SAR_trend_dev_Optical_monthly_std'].loc[:i][-1]:
                curr = merged_pdf_filt3['WA_Optical_cor'].loc[:i][-1]
                mod_val = (merged_pdf_filt3['WA_Optical_cor'].loc[:i][-2] + merged_pdf_filt3['WA_Optical_cor'].loc[:i][-3])/2*(1+sar_trend_weekly_av)
                if pd.isna(mod_val):
                    mod_val = curr
                merged_pdf_filt3['WA_Optical_cor'].loc[:i][-1] = mod_val
                merged_pdf_filt3['WA_Optical_cor_trend'] =  merged_pdf_filt3['WA_Optical_cor'].pct_change()
                    
        merged_pdf_filt3['WA_mean_corr'] = merged_pdf_filt3['WA_Optical_cor']
        final_sa = pd.DataFrame(merged_pdf_filt3['WA_mean_corr'])
        final_sa['area'] = final_sa['WA_mean_corr']
        final_sa.drop(columns=['WA_mean_corr'], inplace = True)
        final_sa['days_passed'] = final_sa.index.to_series().diff().dt.days.fillna(0).astype(int)
        final_sa['name'] = res_name

        data.append(final_sa)

    return pd.concat(data)