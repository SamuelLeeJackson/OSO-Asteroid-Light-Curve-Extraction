import glob
import os
import argparse
import numpy as np
import pandas as pd
import subprocess
import shutil

from scipy.optimize import curve_fit
from scipy import odr
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy.table import Table
from astropy.wcs import utils, WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from astropy.visualization import ZScaleInterval, ImageNormalize, MinMaxInterval, PowerStretch

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import warnings
from tqdm import tqdm

from panstarrs_api import *
import lightCurve as LC
import parameters as pm


parser = argparse.ArgumentParser(
    description='Use of DataList.py, reduces astronomical images for PIRATE.'
)

parser.add_argument(
    '-obj', '--object',
    default=None,
    help='Target object'
)

parser.add_argument(
    '-d', '--date',
    default=None,
    help='Date of observation'
)

parser.add_argument(
    '-i', '--IGNORE',
    default=None,
    help='Comma separated list of PIRATE frame numbers to ignore'
)
parser.add_argument(
    '-t', '--TELESCOPE',
    default='pirate',
    help='Which telescope is the data from (different CT, CTG, & FoV)'
)
parser.add_argument(
    '-fs', '--FRAMESUMMARY',
    default=False,
    help='Toggle for whether to generate frame summary PDF for diagnostics (default: False)'
)
parser.add_argument(
    '-s', '--SIDEREAL',
    default=False,
    help='Toggle for sidereally tracked observations, so position angle criteria can be turned off for finding reference stars. (default: False)'
)

args = parser.parse_args()
object = args.object
date = args.date
ignore = args.IGNORE
telescope = args.TELESCOPE
fs = args.FRAMESUMMARY
sidereal = args.SIDEREAL

script_folder = pm.script_folder
baseFolder = script_folder + object + '/' + date + '/' + telescope.upper() + '/'

sourceFolder = f"{pm.source_folder_base}/{telescope.upper()}/"
dateFolder = sourceFolder + date + '/'
plateSolvedFramesFolder = dateFolder + 'PlateSolvedFrames/'
cataloguesFolder = dateFolder + 'Catalogues/'

if not os.path.isdir(script_folder + object + '/'):
    os.makedirs(script_folder + object + '/')
if not os.path.isdir(baseFolder):
    os.makedirs(baseFolder)
if not os.path.isdir(baseFolder + 'PlateSolvedFrames/'):
    os.makedirs(baseFolder + 'PlateSolvedFrames/')
if not os.path.isdir(baseFolder + 'Catalogues/'):
    os.makedirs(baseFolder + 'Catalogues/')
if not os.path.isdir(baseFolder + 'Junk/'):
    os.makedirs(baseFolder + 'Junk/')
if not os.path.isdir(baseFolder + 'CheckImages/'):
    os.makedirs(baseFolder + 'CheckImages/')

for filename in glob.glob(os.path.join(plateSolvedFramesFolder, f'*A{object}*.fits')):
    print(f"Copying: {filename}")
    shutil.copy(filename, f"{baseFolder}PlateSolvedFrames/")

for filename in glob.glob(os.path.join(cataloguesFolder, f'*A{object}*.cat')):
    print(f"Copying: {filename}")
    shutil.copy(filename, f"{baseFolder}Catalogues/")


def getList(folder, frames_to_ignore=None):
    # Get all fits images in 'folder'
    images = glob.glob1((folder), '*.fits')
    selected_images = []
    for image in images:
        # If there are frames to ignore
        if frames_to_ignore:
            # Check if image number is not in the list of
            # image to ignore
            if not image.split('_')[1] in frames_to_ignore:
                # Add the frame to the list
                selected_images.append(folder + image)
        else:
            # If no frames to ignore then add them
            # all to the list
            selected_images.append(folder + image)
    selected_images.sort()
    return selected_images
        
# Defines the location of the PIRATE observatory,
# Observatorio del Teide, Tenerife
location = pm.config[telescope.lower()]['location']

if ignore:
    frames_to_ignore = ignore.split(',')
    allFrames = getList(baseFolder + 'PlateSolvedFrames/', frames_to_ignore)
else:
    allFrames = getList(baseFolder + 'PlateSolvedFrames/')

lc = LC.LightCurve(object, date, baseFolder, allFrames, telescope.lower(), sidereal)

if fs:
    lc.make_frame_summary()

# Creates a list of only the frames in the R filter
lc.get_filter_frames("R")

# Creates a list of only the frames in the V filter
lc.get_filter_frames("V")

lc.get_frame_pair_with_lowest_airmass()

# Measures the relative lightcurve for each filter
# Ensures the correction to the zero-point is 
# measured for the R frames
rData, ZP1, ZP1err = lc.get_relative_lightcurve('R')
vData = lc.get_relative_lightcurve('V')

# Gets the JD & filter
vFilterArr = [[x[0], x[15]] for x in vData]
rFilterArr = [[x[0], x[15]] for x in rData]

# Combines the two arrays and sorts according to JD
combinedFilterArr = vFilterArr + rFilterArr
combinedFilterArr.sort(key=lambda x: x[0])

# Gets the filters of the images in the order they were taken
filterMap = [x[1] for x in combinedFilterArr]

print("Instrumental photometry done for both filters, combining...")

# Drops the columns for MAG_AUTO, MAGERR_AUTO, SEPARATION, SHIFT,
# SHIFT_ERR, JD (Julain_date - Julian_Date.min()), & FILTER
vMags = pd.DataFrame(vData).drop([1, 2, 3, 10, 11, 14, 15], axis=1)
rMags = pd.DataFrame(rData).drop([1, 2, 3, 10, 11, 14, 15], axis=1)

# Assigns names to the remaining columns and sorts the data according to Julian Date
vMags.columns = ["JD", "AIRMASS", "deltaRA", "deltaDEC", "EXPTIME", "SNR", "ELLIPTICITY", "RELMAG", "RELMAG_ERR"]
vMags.sort_values('JD', inplace=True)

rMags.columns = ["JD", "AIRMASS", "deltaRA", "deltaDEC", "EXPTIME", "SNR", "ELLIPTICITY", "RELMAG", "RELMAG_ERR"]
rMags.sort_values('JD', inplace=True)

# Plots the offsets of the object from predicted coords
plt.rcParams.update({'font.size': 16})
plt.figure(1, (12, 10))
plt.xlabel("Offset in Right Ascension [arcseconds]")
plt.ylabel("Offset in Declination [arcseconds]")
ax = plt.gca()

plt.scatter(x=vMags.deltaRA, y=vMags.deltaDEC, marker='x',
            color='g', label='V filter')
plt.scatter(x=rMags.deltaRA, y=rMags.deltaDEC, marker='o',
            color='r', label='R filter')
x_lims_max = np.max([abs(x) for x in ax.get_xlim()])
y_lims_max = np.max([abs(y) for y in ax.get_ylim()])
plt.axhline(color='black', lw=0.5)
plt.axvline(color='black', lw=0.5)
ax.set_xlim(-x_lims_max, x_lims_max)
ax.set_ylim(-y_lims_max, y_lims_max)
plt.savefig(baseFolder + "deltaCoords.pdf", bbox_inches='tight')
plt.close()

# All values in the dataframe should be floats by this point
vMags = vMags.astype('float')

# Initialises the colour variables
# Unsure why I needed this but I'm too scared
# to remove it
# instColour = 0
# instColourErr = 0
# psColour = 0
# psColourErr = 0

print("Using measured colours")

# Known from previous colour transformation work
# Measured with a large ensemble of stars

m = lc.CTG
mErr = lc.CTG_err

# Measure the transformation intercept corresponding to the first frames in each filter
# This requires that these frames are near to eachother in time/airmass
# Lone initial frames will need to be ignored for this to work
c, c_err = lc.fit_pair_transformation_intercept()

# Get the average instrumental (V - R) colour
# Also keeps the dataframe with each individual value for conversion
instColour, instColourErr = lc.get_colour(rMags, vMags)

# Convert each instrumental colour into a Pan-STARRS colour
psColour = instColour * m + c

# Calculates the uncertainty on the Pan-STARRS colour
# TODO: Tie an equation number to this formula at some point
psColourErr = np.sqrt(
    np.square(c_err) + 
    np.square(instColour * m) *
    (np.square(instColourErr / instColour) +
    np.square(mErr/m))
)

# Correct the V magnitude relative lightcurve to the R-band
vMags['RELMAG'] = vMags['RELMAG'] - instColour
rMags['RELMAG_DISP_ERR'] = rMags['RELMAG_ERR']
vMags['RELMAG_DISP_ERR'] = vMags['RELMAG_ERR']
vMags['RELMAG_ERR'] = np.sqrt(
    np.square(vMags['RELMAG_ERR']) + np.square(instColourErr)
)

# Create a DataFrame containing the both relative lightcurves 
# (V corrected to the R-band using the instrumental colour)
df = pd.concat([rMags, vMags], ignore_index=True, sort=False)
df.sort_values('JD', inplace=True)
df = df.astype('float')

if '_' in object:
    asteroid = object.replace('_', ' ')
else:
    asteroid = object

# Measure average heliocentric distance, geocentric distance, light travel time, and phase angle for the phase curve
r, delta, LTT, alpha, range_unc, rate_of_motion = lc.get_geometry_horizons(df['JD'].mean())
df['alpha'] = alpha
df['r'] = r
df['delta'] = delta
df['LTT'] = LTT
df['range_unc'] = range_unc
df['Rate_of_Motion'] = rate_of_motion

# Create 'Not Light Time Corrected' JD column
df['Julian_Date_NLTC'] = df['JD']

# Create 'Light Time Corrected' JD column
df['Julian_Date_LTC'] = df['JD'] - LTT

# Create JD column which just measures time in days since first frame
minJD = df['Julian_Date_LTC'].min()
df['JD'] = df['Julian_Date_LTC'] - minJD

# Plot the relative lightcurve
plt.rcParams.update({'font.size': 20})
plt.figure(1, (14, 8))
plt.errorbar(x=df['JD'], y=df['RELMAG'], yerr=df['RELMAG_DISP_ERR'], fmt='o',
             markerfacecolor='none')
plt.xlabel(f'Julian Date - {minJD}')
plt.ylabel('Shifted (Relative) Magnitudes')
plt.gca().invert_yaxis()
plt.savefig(
    baseFolder + f'RelativeLightcurve.pdf', bbox_inches='tight'
)
plt.close()

# Define constants to be used in conversion equations
ZP = ZP1
ZPerr = ZP1err

# Calculate CT * (g_P1 - r_P1)
# Pre-calculate the error on this part of the calculation
CTpsColour = lc.CT * psColour
CTpsColourErr = CTpsColour * np.sqrt(
    np.square(lc.CTerr/lc.CT) + np.square(psColourErr/psColour)
)

# Convert relative lightcurve to Pan-STARRS r-band using the zero-point and colour term
df['r_P1'] = df['RELMAG'] - ZP - CTpsColour

# Calculate the uncertainty on the r_P1 lightcurve points
df['r_P1_err'] = np.sqrt(
    np.square(df['RELMAG_ERR']) + np.square(ZPerr) + np.square(CTpsColourErr)
)

# Plot the r_P1 lightcurve
plt.rcParams.update({'font.size': 16})
plt.figure(1, (12, 8))
plt.errorbar(x=df['JD'], y=df['r_P1'], yerr=df['r_P1_err'], fmt='o',
             markerfacecolor='none')
plt.xlabel(f'Julian Date - {minJD}')
plt.ylabel('Calibrated Magnitude [Pan-STARRS r-band]')
plt.gca().invert_yaxis()
plt.savefig(baseFolder + 'CalibratedLightcurve.pdf', bbox_inches='tight')
plt.close()

# Calculate the (B - V) colour from the Pan-STARRS colour
# Using a relation and corresponding uncertainties on the params from Tonry et al. (2012)
topLine = psColour + 0.181
BVcolour = topLine/0.892

# Calculate the uncertainty on the (B - V) colour 
# using standard propagation of uncertainty formulae
topLineErr = np.sqrt(np.square(psColourErr) + np.square(0.028))
BVcolourErr = BVcolour * np.sqrt(
    np.square(0.028/0.892) + np.square(topLineErr/topLine)
)

# Calculate the last term (& uncertainty) in the colour conversion equation
# V = r_P1 - 0.082 + 0.462 * (B - V) - 0.041 * (B - V)**2
lastTerm = 0.041*np.square(BVcolour)
lastTermErr = lastTerm * np.sqrt(
    np.square(0.025/0.041) + 2 * np.square(BVcolourErr/BVcolour)
)

# Calculate the third term (& uncertainty) in the colour conversion equation
# V = r_P1 - 0.082 + 0.462 * (B - V) - 0.041 * (B - V)**2
middleTerm = 0.462*BVcolour
middleTermErr = middleTerm * np.sqrt(
    np.square(0.025/0.462) + np.square(BVcolourErr/BVcolour)
)

# V = r_P1 - 0.082 + 0.462 * (B - V) - 0.041 * (B - V)**2
# Then calculate V magnitude uncertainty using standard propogation formulae
df['Vmag'] = df['r_P1'] - 0.082 + middleTerm - lastTerm
df['Vmag_err'] = np.sqrt(
    np.square(df['r_P1_err']) +
    np.square(0.025) +
    np.square(middleTermErr) +
    np.square(lastTermErr)
)

# Plot the V magnitude lightcurve
plt.rcParams.update({'font.size': 16})
plt.figure(1, (12, 8))
plt.errorbar(x=df['JD'], y=df['Vmag'], yerr=df['Vmag_err'], fmt='o',
             markerfacecolor='none')
plt.xlabel(f'Julian Date - {minJD}')
plt.ylabel('$V(r, \\Delta, \\alpha)$')
plt.gca().invert_yaxis()
plt.savefig(baseFolder + 'VbandLightcurve.pdf', bbox_inches='tight')
plt.close()

# Calculate the reduced magnitude
# No uncertainty propagation since r and delta are 
# considered as having no uncertainty from HORIZONS

df['REDUCED_MAG'] = df['Vmag'] - 5*np.log10(r * delta)

r_delta_unc = r*delta * np.sqrt(np.square(range_unc/r) + np.square(range_unc/delta))
log_r_delta_unc = 0.434 * (r_delta_unc / r*delta)
five_log_r_delta_unc = 5 * log_r_delta_unc

df['REDUCED_MAG_UNC'] = np.sqrt(np.square(df.Vmag_err) + np.square(five_log_r_delta_unc))

df['Scale_Unc'] = np.sqrt(
    np.square(ZPerr) + 
    np.square(CTpsColourErr) +
    np.square(0.025) +
    np.square(middleTermErr) +
    np.square(lastTermErr) + 
    np.square(five_log_r_delta_unc)
)

scaleUnc = df.Scale_Unc.mean()

# Plot the reduced magnitude lightcurve
plt.rcParams.update({'font.size': 20})
plt.figure(1, (14, 8))
plt.errorbar(x=df['JD'], y=df['REDUCED_MAG'], yerr=df['REDUCED_MAG_UNC'], fmt='o',
             markerfacecolor='none')
plt.xlabel(f'Julian Date - {minJD}')
plt.ylabel(f'$V(1, 1, {float(alpha):.2f})$')
plt.gca().invert_yaxis()
plt.savefig(baseFolder + 'ReducedVbandLightcurve.pdf', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(14, 10))
spec = gridspec.GridSpec(1, 2, width_ratios=[4, .25], wspace=0.0)


ax0 = fig.add_subplot(spec[0])
ax0.invert_yaxis()
ln0 = ax0.errorbar(x=df['JD'], y=df['REDUCED_MAG'], yerr=df['RELMAG_DISP_ERR'], fmt='o', markerfacecolor='none', elinewidth=1, c='k', label='Reduced V Magnitude Light Curve')
ax0.set_xlabel(f'Julian Date - {minJD}')
ax0.set_ylabel(f'$V(1 $ AU$, 1 $ AU$, {float(alpha):.2f})$ [Johnson V mag]')

mean = df.REDUCED_MAG.mean()

ylims = ax0.get_ylim()
y_range = np.diff(ylims)
scale_unc_y = ylims[1] + scaleUnc - 0.05 * y_range

xlims = ax0.get_xlim()
x_range = np.diff(xlims)
scale_unc_x = xlims[0] + 0.05 * x_range


ax1 = fig.add_subplot(spec[1])
ax1.invert_yaxis()

ax1.set_xlim(*ax0.get_xlim())
ax1.set_ylim([x - mean for x in ax0.get_ylim()])

xlims = ax1.get_xlim()
x_range = np.diff(xlims)
scale_unc_x = xlims[0] + 0.5 * x_range

ln1 = ax1.errorbar(scale_unc_x, 0, yerr=scaleUnc, fmt='none', c='b', elinewidth=2, capthick=2, capsize=6, markerfacecolor='none', label='Scale Uncertainty')
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position("right")
ax1.set_ylabel('Scale Uncertainty [Johnson V mag]')
ax1.set_xticks([])

lns = [ln0, ln1]
labs = [l.get_label() for l in lns]
ax0.legend(lns, labs, loc='best')
plt.savefig(baseFolder + 'ReducedVbandLightcurve_newUnc.pdf', bbox_inches='tight')
plt.close()

# Add the filter map to the dataframe
df['Filter'] = filterMap

# Plot the signal-to-noise of each measurement, separated by filter
# as a function of JD
plt.rcParams.update({'font.size': 16})
plt.figure(1, (12, 10))
plt.xlabel(f'Julian Date - {minJD}')
plt.ylabel('Signal to Noise Ratio')
plt.scatter(df['JD'][df.Filter == 'R'], df['SNR'][df.Filter == 'R'], c='r', label='R-band')
plt.scatter(df['JD'][df.Filter == 'V'], df['SNR'][df.Filter == 'V'], c='g', label='V-band')
plt.savefig(baseFolder + 'SNR_JD.pdf', bbox_inches='tight')
plt.close()

# Plot the signal-to-noise of each measurement, separated by filter
# as a function of airmass
plt.rcParams.update({'font.size': 16})
plt.figure(1, (12, 10))
plt.xlabel(f'Julian Date - {minJD}')
plt.ylabel('Signal to Noise Ratio')
plt.scatter(df['AIRMASS'][df.Filter == 'R'], df['SNR'][df.Filter == 'R'], c='r', label='R-band')
plt.scatter(df['AIRMASS'][df.Filter == 'V'], df['SNR'][df.Filter == 'V'], c='g', label='V-band')
plt.savefig(baseFolder + 'SNR_AIRMASS.pdf', bbox_inches='tight')
plt.close()

# Add all information and calculated values to the dataframe before it is written to a file
# This should enable checking calculations outside of the script
df['instColour'] = instColour
df['instColourErr'] = instColourErr
df['psColour'] = psColour
df['psColourErr'] = psColourErr
df['BVcolour'] = BVcolour
df['BVcolourErr']  = BVcolourErr
df['CT'] = lc.CT
df['CTerr'] = lc.CTerr
df['ZP'] = ZP
df['ZPerr'] = ZPerr
df['m'] = m
df['m_err'] = mErr
df['c'] = c
df['c_err'] = c_err

# Write the data to a file
df.to_csv(baseFolder + f'{object}-{date}.csv', index=False)

# Remove the folders containing images and catalogues once finished to save disk space
shutil.rmtree(baseFolder + 'PlateSolvedFrames/')
shutil.rmtree(baseFolder + 'Catalogues/')
shutil.rmtree(baseFolder + 'Junk/')
shutil.rmtree(baseFolder + 'CheckImages/')

print("Light Curve Extraction Completed")