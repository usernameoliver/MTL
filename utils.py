import nibabel as nib
import SimpleITK as sitk
import numpy as np
from sksurv.metrics import concordance_index_censored

def count_live_years(series):
    index = 0
    while index < len(series) and abs(series[index] - 1) < 0.01:
        index += 1
    return index - 1

def calculate_concordance_index(predictions, groundtruth):
    predicted_risk = []
    gt_years = []
    event = []
    num_patient, num_tasks = np.shape(predictions)
    for i in range(num_patient):
        gt_years.append(count_live_years(groundtruth[i]) + 1)
        if count_live_years(predictions[i]) + 1 == 0:
            predicted_risk.append(10)
        else:
            predicted_risk.append(1.0 / ( count_live_years(predictions[i]) + 1))
        if groundtruth[i].count(-1) > 0:
            event.append(False)
        else:
            event.append(True)
    event_status = np.array(event)
    event_time = np.array(gt_years)
    predicted_risk = np.array(predicted_risk)
    c_index2 = concordance_index_censored(event_status, event_time, predicted_risk)
    return c_index2[0]

def save_nifti(mask_preds, pids):

    num = len(pids)
    for i in range(num):
        pid = pids[i]
        mask_pred = mask_preds[i]
        mask_pred = np.squeeze(mask_pred) # drop singleton dim in case temporal dim exists
        top = np.max(mask_pred)
        bottom = np.min(mask_pred)
        mask_pred = 255.0 * (mask_pred - bottom) / (top - bottom)
        outputImageFileName = 'seg_result/' + str(pid) + '.nii'
        out = sitk.GetImageFromArray(mask_pred)
        sitk.WriteImage(out,outputImageFileName)

def load_nifti_img(filepath, dtype):
    '''
        NIFTI Image Loader
        :param filepath: path to the input NIFTI image
        :param dtype: dataio type of the nifti numpy array
        :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    return out_nii_array
