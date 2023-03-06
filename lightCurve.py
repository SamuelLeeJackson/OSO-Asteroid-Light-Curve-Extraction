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
import matplotlib.patches as patches
from astropy.visualization import ZScaleInterval, ImageNormalize, MinMaxInterval, PowerStretch

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import warnings
from tqdm import tqdm

import parameters as pm
from panstarrs_api import *

class LightCurve:
    def __init__(self, object, date, baseFolder, frames, telescope, sidereal):
        self.object = object
        self.date = date
        self.baseFolder = baseFolder
        self.fits_images = frames
        self.telescope = telescope
        self.sidereal = sidereal


        self.filter_frame_lists = {
            'V': [],
            'R': []
        }

        self.best_filter_frames = {
            'V': 0,
            'R': 0
        }

        self.CT = pm.config[telescope]['CT']
        self.CTerr = pm.config[telescope]['CTerr']
        self.CTG = pm.config[telescope]['CTG']
        self.CTG_err = pm.config[telescope]['CTG_err']
        self.FoV = pm.config[telescope]['FoV']
        self.location = pm.config[telescope]['location']

    # timing decorator, to use this put "@time_dec" in front of any def
    # (like the @staticmethods below) and it should print
    # how long it took to use that def.
    def time_dec(func):
        def wrapper(*args):
            t = datetime.datetime.now()
            res = func(*args)
            logging.debug("{}: {}".format(func.__name__, datetime.datetime.now()-t))
            return res
        return wrapper

    def make_frame_summary(self):
        for frame in self.fits_images:
            data = fits.getdata(frame)
            norm = ImageNormalize(data, interval=ZScaleInterval())
            vmin = norm.vmin
            vmax = norm.vmax
            plt.imsave(self.baseFolder + f"{frame.split('/')[-1]}.png", format='png', origin='lower', cmap='gray', arr=data, vmin=vmin, vmax=vmax)
        subprocess.call(f'convert {self.baseFolder}{self.telescope.upper()}_*.fits.png {self.baseFolder}AllFrames_Summary.pdf', shell=True)
        subprocess.call(f'rm {self.baseFolder}{self.telescope.upper()}_*.fits.png', shell=True)
    
    
    def get_filter_frames(self, filter):
        
        # Loop over each image in the list
        # and check if the filter is 'filter'
        # If so, add to the list
        frameList = []
        for frame in self.fits_images:
            header = fits.getheader(frame)
            if header['FILTER'] == filter:
                frameList.append(frame)
                    
        frameList.sort()
        self.filter_frame_lists[filter] = frameList


    def get_frame_pair_with_lowest_airmass(self):

        rFrames = self.filter_frame_lists['R']
        vFrames = self.filter_frame_lists['V']
        
        # Create a list of frames with their Julian Date
        arrR = []
        for rFrame in rFrames:
            header = fits.getheader(rFrame)
            arrR.append([rFrame, header['JD'], float(header['ALTITUDE'])])
        arrR = pd.DataFrame(arrR, columns=["frame", "JD", "Alt"])

        # For each V frame, find the nearest R frame by Julian Date
        pairs = []
        for vFrame in vFrames:
            header = fits.getheader(vFrame)
            JDdiff = abs(arrR.JD - header['JD'])
            arrR['JDdiff'] = JDdiff
            nearestR = arrR[arrR.JDdiff == JDdiff.min()]['frame'].values[0]
            nearestR_altitude = arrR[arrR.JDdiff == JDdiff.min()]['Alt'].values[0]
            mean_altitude = (float(header['ALTITUDE']) + nearestR_altitude) / 2
            pairs.append([vFrame, nearestR, mean_altitude])

        pairs_dataframe = pd.DataFrame(pairs, columns=["v_frame", "r_frame", "mean_alt"])
        pair = pairs_dataframe[pairs_dataframe.mean_alt == pairs_dataframe.mean_alt.max()].values[0]
        best_v = pair[0]
        best_r = pair[1]
        
        R_index = rFrames.index(best_r)
        V_index = vFrames.index(best_v)
            
        self.best_filter_frames['R'] = R_index
        self.best_filter_frames['V'] = V_index

    def get_relative_lightcurve(self, filt):
        # Define the reference frame for the relative lightcurve
        frames = self.filter_frame_lists[filt]
        reference_index = self.best_filter_frames[filt]
        refFrame = frames[reference_index]

        # Get the sextractor catalogue for the reference frame
        sex_refcat = (
            self.baseFolder +
            'Catalogues/' +
            refFrame.split('/')[-1].strip('.fits') +
            '.cat'
        )

        # For each image, find stars in the catalogue
        # that match stars in the reference catalogue
        # Creates an array of dataframes (1 dataframe per image)
        dfArr = []
        for frame in frames:
            df = self.find_refstars(frame, sex_refcat)
            print(len(df))
            dfArr.append(df)

        refStars = []
        # For each star in the reference frame
        for idx, row in dfArr[reference_index].iterrows():
            count = 0
            # Look in each dataframe
            for df in dfArr:
                # If the reference frame star is in this dataframe, add to a count
                if row['ra'] in df['ra'].values:
                    count = count+1
                    
            # If the star has been found in every frame, then it can be used as a reference star
            if count == len(frames):
                # Add it to the list of reference stars
                refStars.append([row['ra'], row['dec']])
                
        

        refData = []
        numRefStars = 0
        # For each reference star, get the full data from each catalogues
        for ref in refStars:
            count = 0
            for df in dfArr:
                if len(df[(df['ra'] == ref[0]) & (df['dec'] == ref[1])].values) == 1:
                    count += 1
            if count == len(dfArr):
                numRefStars += 1
                for df in dfArr:
                    refData.append(
                        df[(df['ra'] == ref[0]) & (df['dec'] == ref[1])].values[0]
                    )
            else:
                continue
                
        print(f"Number of usable reference stars: {numRefStars}")
        
        pd.options.mode.chained_assignment = None
        refStars = pd.DataFrame(
            refData, columns=[
                "RefRA", "RefDEC", "RA_frame", "DEC_frame",
                "JD", "MAG_AUTO", "MAGERR_AUTO"
            ]
        )
        refStars = refStars.astype('float64')
        
        refStars['CoordIdentifier'] = refStars['RefRA'].astype(str) + refStars['RefDEC'].astype(str)
        # Plot the shift in brightness of each reference star as a function of time
        plt.rcParams.update({'font.size': 16})
        plt.figure(1, (12, 8))
        i = 0
        for CoordIdentifier in refStars['CoordIdentifier'].unique():
            i = i + 1
            unique = refStars[refStars['CoordIdentifier'] == CoordIdentifier]
            unique['JD'] = unique['JD']
            unique['JD'] = unique['JD'] - unique['JD'].min()
            unique['MAG_AUTO'] = unique['MAG_AUTO']
            unique['MAG_AUTO'] = unique['MAG_AUTO'] - unique['MAG_AUTO'].iloc[reference_index]
            unique['MAGERR_AUTO'] = unique['MAGERR_AUTO']
            plt.errorbar(
                unique['JD'], unique['MAG_AUTO'], unique['MAGERR_AUTO'], fmt='o'
            )
        jdMin = refStars['JD'].min()
        plt.xlabel(f'JD - {jdMin}')
        plt.ylabel('Inst. Mag')
        plt.gca().invert_yaxis()
        plt.savefig(self.baseFolder + f'stars_{filt}.pdf', bbox_inches='tight')
        plt.close()


        noRefStars = 0
        # Create a photometry dataframe, using the Julian Dates of the reference star measurements
        phot_df = pd.DataFrame(refStars['JD'].unique(), columns=["JD"])
        
        for CoordIdentifier in refStars['CoordIdentifier'].unique():
            # For each reference star, add to a counter
            noRefStars = noRefStars + 1
            # Get the data for just this star
            unique = refStars[refStars['CoordIdentifier'] == CoordIdentifier]
            # Create a shift column for this reference star
            phot_df[f'Star{noRefStars}_shift'] = (
                unique['MAG_AUTO'] - unique['MAG_AUTO'].iloc[reference_index]
            ).values
            
            # Create a shift uncertainty column for this reference star
            phot_df[f'Star{noRefStars}_err'] = unique['MAGERR_AUTO'].values

        refStars.drop(columns='CoordIdentifier')
        
        wAveTopLine = []
        wAveBottomLine = []
        
        for i in range(noRefStars):
            # For each reference star
            
            # Star number was 1-indexed so need to add one to the iterator value
            i = i + 1
            
            # Build up the top line of the weighted average sum
            wAveTopLine.append(
                phot_df[f'Star{i}_shift'].values/np.square(
                    phot_df[f'Star{i}_err'].values
                )
            )
            
            # Build up the bottom line of the weighted average sum
            wAveBottomLine.append(1/np.square(phot_df[f'Star{i}_err'].values))
            
        # Calculate the top & bottom lines of the weighted average sum
        topLine = pd.DataFrame(wAveTopLine).transpose().sum(axis=1)
        botLine = pd.DataFrame(wAveBottomLine).transpose().sum(axis=1)
        
        # Calculate the weighted average shifts and uncertainty
        weightedShifts = topLine/botLine
        weightedShiftsErr = np.sqrt(1/botLine)

        # Get the instrumental photometry of the asteroid
        inst_data = []
        for frame in frames:
            data = self.inst_phot(frame)
            inst_data.append(data)

        df = pd.DataFrame(
            inst_data, columns=[
                "Julian_Date", "MAG_AUTO", "MAGERR_AUTO", "SEPARATION", "AIRMASS",
                "deltaRA", "deltaDEC", "EXPTIME", "SNR", "ELLIPTICITY"
            ]
        )


        df['SHIFT'] = weightedShifts
        df['SHIFT_ERR'] = weightedShiftsErr
        
        # Correct the instrumental photometry to a relative lightcurve
        df['RELMAG'] = df['MAG_AUTO'] - df['SHIFT']
        
        # Propagate the uncertainty
        df['RELMAG_ERR'] = np.sqrt(
            np.square(df['MAGERR_AUTO']) + np.square(df['SHIFT_ERR'])
        )

        # Column representing days since first frame
        minJD = df['Julian_Date'].min()
        df['JD'] = df['Julian_Date'] - minJD

        # Remove any datapoints where the asteroid is more than 10 arcsecodns from where it is expected
        df = df[df['SEPARATION'] < 20]

        # Plot the instrumental lightcurve
        plt.rcParams.update({'font.size': 16})
        plt.figure(1, (12, 8))
        plt.errorbar(
            x=df['JD'], y=df['MAG_AUTO'], yerr=df['MAGERR_AUTO'], fmt='o',
            markerfacecolor='none'
        )
        plt.xlabel(f'Julian Date - {minJD}')
        plt.ylabel('Inst. Mag')
        plt.gca().invert_yaxis()
        plt.savefig(
            self.baseFolder + f'InstrumentalLightcurve_{filt}.pdf',
            bbox_inches='tight'
        )
        plt.close()

        # Plot the offset of the asteroid from the predicted coordinate as a function of time
        plt.rcParams.update({'font.size': 16})
        plt.figure(1, (12, 8))
        plt.scatter(x=df['JD'], y=df['SEPARATION'], marker='o', c='k')
        plt.xlabel(f'Julian Date - {minJD}')
        plt.ylabel('Asteroid Position Offset [arcseconds]')
        plt.savefig(
            self.baseFolder + f'AsteroidOffset_{filt}.pdf',
            bbox_inches='tight'
        )
        plt.close()

        # Plot the deltas as a scatter
        plt.rcParams.update({'font.size': 20})
        plt.figure(1, (12, 10))
        plt.scatter(x=df['deltaRA'], y=df['deltaDEC'], marker='x', c='k')
        plt.xlabel('Offset in Right Ascension [arcseconds]')
        plt.ylabel('Offset in Declination [arcseconds]')
        ax = plt.gca()
        x_lims_max = np.max([abs(x) for x in ax.get_xlim()])
        y_lims_max = np.max([abs(y) for y in ax.get_ylim()])
        plt.axhline(color='black', lw=0.5)
        plt.axvline(color='black', lw=0.5)
        ax.set_xlim(-x_lims_max, x_lims_max)
        ax.set_ylim(-y_lims_max, y_lims_max)
        plt.savefig(
            self.baseFolder + f'deltaCoords_{filt}.pdf', bbox_inches='tight'
        )
        plt.close()

        # Plot the relative lightcurve
        plt.rcParams.update({'font.size': 16})
        plt.figure(1, (12, 8))
        plt.errorbar(x=df['JD'], y=df['RELMAG'], yerr=df['RELMAG_ERR'], fmt='o',
                    markerfacecolor='none')
        plt.xlabel(f'Julian Date - {minJD}')
        plt.ylabel('Rel. Mag')
        plt.gca().invert_yaxis()
        plt.savefig(
            self.baseFolder + f'RelativeLightcurve_{filt}.pdf',
            bbox_inches='tight'
        )
        plt.close()

        # Plot the combined instrumental and relative lightcurve
        plt.rcParams.update({'font.size': 20})
        plt.figure(1, (14, 8))
        plt.errorbar(
            x=df['JD'], y=df['MAG_AUTO'], yerr=df['MAGERR_AUTO'], fmt='o',
            markerfacecolor='none', ecolor='r', mec='r', label='Unshifted Light Curve'
        )
        plt.errorbar(x=df['JD'], y=df['RELMAG'], yerr=df['RELMAG_ERR'], fmt='o',
                    markerfacecolor='none', ecolor='b', mec='b', label='Relative (shifted) Light Curve')
        plt.xlabel(f'Julian Date - {minJD}')
        plt.ylabel('Instrumental R Magnitudes')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(
            self.baseFolder + f'CombinedLightcurve_{filt}.pdf',
            bbox_inches='tight'
        )
        plt.close()

        # Plot the weighted shift averages of the stars as a function of time
        plt.rcParams.update({'font.size': 20})
        plt.figure(1, (10, 8))
        plt.errorbar(x=df['JD'], y=df['SHIFT'], yerr=df['SHIFT_ERR'], fmt='o',
                    markerfacecolor='none')
        plt.xlabel(f'Julian Date - {minJD}')
        plt.ylabel('Magnitude Shift from Reference Frame')
        plt.gca().invert_yaxis()
        plt.savefig(self.baseFolder + f'ShiftCurveJD_{filt}.pdf', bbox_inches='tight')
        plt.close()

        # PLot the weighted shift averages of the stars as a function of airmass
        plt.rcParams.update({'font.size': 20})
        plt.figure(1, (10, 8))
        plt.errorbar(
            x=df['AIRMASS'].astype(float), y=df['SHIFT'],
            yerr=df['SHIFT_ERR'], fmt='o', markerfacecolor='none'
        )
        plt.xlabel('Airmass')
        plt.ylabel('Magnitude Shift from Reference Frame')
        plt.gca().invert_yaxis()
        plt.savefig(
            self.baseFolder + f'ShiftCurveAirmass_{filt}.pdf',
            bbox_inches='tight'
        )
        plt.close()

        # Add the filter to the returned values
        df['FILTER'] = filt
        
        if filt == 'R':
        
            # Assuming no errors, the change in the zero points from frame to frame should
            # match the weighted shift average
            
            # By fitting a linear relation, and measuring the offset from a 1:1 relation (c)
            # we estimate the error correction needed on the first zero point
        
            # Measure the photometric zero-point w.r.t the PS1 system
            # for all R-band images
            ZParr = self.get_ps1_stars_all('R')
    
            def linear(x, c):
                return x + c
                
            ZP_arr2 = ZParr[:, 0] - weightedShifts
            ZP1, ZP1_median, ZP1err = sigma_clipped_stats(ZP_arr2)
    #         ZP1 = np.mean(ZP_arr2)
    #         ZP1err = np.std(ZP_arr2) # / np.sqrt(len(ZP_arr2))
            
            plt.rcParams.update({'font.size': 16})
            plt.figure(1, (12, 10))
            plt.errorbar(range(1, len(ZP_arr2) + 1, 1), ZP_arr2, yerr=ZParr[:, 1], fmt='o', markerfacecolor='none', color='k')
            plt.axhline(ZP1, c='r', linestyle='--')
            plt.axhspan(ZP1-ZP1err, ZP1+ZP1err, color='r', alpha=.25)
            plt.xlabel('Frame')
            plt.ylabel('Zero Point')
            plt.savefig(self.baseFolder + 'ZeroPointsDetrended.pdf', bbox_inches='tight')
            plt.close()
            
            return df.values, ZP1, ZP1err
        
        return df.values

    def find_refstars(self, frame, sex_refcat):
        print(f"Finding refstars: {frame}")
        # Define naming convention for files using base name of image
        baseName = frame.strip('.fits')
        # Define catalogue name from pipeline
        catalogueName = (
            self.baseFolder +
            'Catalogues/' +
            baseName.split('/')[-1] +
            '.cat'
        )
        # PIRATE data commonly has non-standard fits keywords so 'silentfix'
        # helps to resolve this
        image = fits.open(frame)
        header = image[0].header
        jd = header['JD']
        exptime = header['EXPTIME']
        # Read in the source extractor catalogue and filter it as follows:
        #   Objects must have a FWHM greater than one pixel (removes artifacts)
        #   Objects can't have flags indicating photometry errors
        #   Area of object in the image must be greater than six pixels

        catalogue = ascii.read(catalogueName, format='sextractor')
        catalogue = catalogue[(catalogue['FWHM_IMAGE'] > 1) &
                            (catalogue['FLAGS'] == 0) &
                            (catalogue['ISOAREA_IMAGE'] > 6) &
                            (catalogue['FLUX_MAX'] < 55000)]
        catalogue.sort('MAG_AUTO')
        catalogue.remove_rows(slice(200, len(catalogue)))

        ref_catalogue = ascii.read(sex_refcat, format='sextractor')
        ref_catalogue = ref_catalogue[
            (ref_catalogue['FWHM_IMAGE'] > 1) &
            (ref_catalogue['FLAGS'] == 0) &
            (ref_catalogue['ISOAREA_IMAGE'] > 6) &
            (ref_catalogue['FLUX_MAX'] < 55000)
        ]

        ref_catalogue.sort('MAG_AUTO')
        ref_catalogue.remove_rows(slice(200, len(ref_catalogue)))
        # Find the position angle of each source and determine the sigma-clipped
        # mean position angle. With trailed images all of the stars should be
        # trailed in the same direction. We therefore ignore all objects with
        # position angles more than 3-sigma away from the mean.

        theta = catalogue['THETA_IMAGE']
        mean, median, std = sigma_clipped_stats(theta, cenfunc='mean')
        if not self.sidereal:
            catalogue = catalogue[abs(catalogue['THETA_IMAGE']-mean) < 3*std]

        theta = ref_catalogue['THETA_IMAGE']
        mean, median, std = sigma_clipped_stats(theta, cenfunc='mean')
        if not self.sidereal:
            ref_catalogue = ref_catalogue[abs(ref_catalogue['THETA_IMAGE']-mean) < 3*std]

        # Convert Astropy ascii tables to pandas for easier manipulation and loop
        # over all rows in the dataframe representing the PanStarrs catalogue
        pdCatalogue = catalogue.to_pandas()
        pdRefCatalogue = ref_catalogue.to_pandas()
        arr = []
        for index, row in pdRefCatalogue.iterrows():
            # Read in RA & DEC of object in PS1 catalogue
            # and set astrometric precision at 2 arcseconds
            ra = row['ALPHA_J2000']
            dec = row['DELTA_J2000']
            precision = 5.0/3600.0

            # Return entries in the sextractor catalogue within the astrometric
            # precision of the selected star from the PanStarrs catalogue
            catalogueEntryMatch = pdCatalogue[
                (abs(pdCatalogue['ALPHA_J2000'] - ra) < precision) &
                (abs(pdCatalogue['DELTA_J2000'] - dec) < precision)
            ]

            # If a unique star is found within the astrometric precision then it is
            # taken to be a match within the catalogues
            if len(catalogueEntryMatch) == 1:
                # Extract the relevant measured paramters from the sextractor
                # catalogue for the matched star
                flux = catalogueEntryMatch['FLUX_AUTO'].values[0]
                mag = -2.5 * np.log10(flux/exptime)
                mag_err = catalogueEntryMatch['MAGERR_AUTO'].values[0]
                ALPHA_J2000 = catalogueEntryMatch['ALPHA_J2000'].values[0]
                DELTA_J2000 = catalogueEntryMatch['DELTA_J2000'].values[0]

                # Append the required information to the array of matched stars
                arr.append([ra, dec, ALPHA_J2000, DELTA_J2000,
                            jd, mag, mag_err])

        # Convert array to a dataframe for ease of manipulation
        df = pd.DataFrame(arr, columns=['ra', 'dec', 'ALPHA_J2000', 'DELTA_J2000',
                                        'JD', 'R_frame', 'R_frame_err'])

        return df


    def get_horizons_info(self, julian_date):

        horizonsString = (
            "https://ssd.jpl.nasa.gov/api/horizons.api?format=text" +
            f"&COMMAND='{self.object}%3B'" +  # %3B represents a semi-colon (small body)
            "&CENTER='coord@399'" +
            "&COORD_TYPE='GEODETIC'" +
            "&SITE_COORD='" +
            f"{self.location['lon']},{self.location['lat']},{self.location['elevation']}'" +
            "&MAKE_EPHEM='YES'" +
            "&TABLE_TYPE='OBSERVER'" +
            f"&TLIST='{julian_date}'" +
            "&CAL_FORMAT='CAL'" +
            "&TIME_DIGITS='SECONDS'" +
            "&ANG_FORMAT='DEG'" +
            "&OUT_UNITS='KM-S'" +
            "&RANGE_UNITS='AU'" +
            "&APPARENT='AIRLESS'" +
            "&SUPPRESS_RANGE_RATE='NO'" +
            "&SKIP_DAYLT='NO'" +
            "&EXTRA_PREC='YES'" +
            "&R_T_S_ONLY='NO'" +
            "&REF_SYSTEM='J2000'" +
            "&CSV_FORMAT='YES'" +
            "&OBJ_DATA='NO'" +
            "&QUANTITIES='1,8'"
        )
        response = requests.get(horizonsString)
        data = response.text
        start = "$$SOE"
        end = "$$EOE"
        ephem = data[data.find(start)+len(start):data.rfind(end)]
        ephem = ephem.replace(" ", "").replace("\n", "").split(',')

        # Return J2000 RA & DEC, Airmass
        return ephem[3], ephem[4], ephem[5]


    def inst_phot(self, frame):
        # Define naming convention for files using base name of image
        baseName = frame.strip('.fits')
        # Define catalogue name from pipeline
        catalogueName = self.baseFolder + 'Catalogues/' + baseName.split('/')[-1] + '.cat'

        # PIRATE data commonly has non-standard fits keywords so 'silentfix'
        # helps to resolve this
        image = fits.open(frame)
        header = image[0].header

        # Extracts the World Coordinate System info from the header
        # (Image should be plate solved from the pipeline
        wcs = WCS(header)

        # Determines the size of the image in pixels
        ccd_x = header['IMAGEW']
        ccd_y = header['IMAGEH']
        jd = header['JD']
        exptime = header['EXPTIME']

        # Converts the centre pixel to RA & DEC (origin=1 required for FITS
        # images)
        frameCentre = utils.pixel_to_skycoord(ccd_x//2, ccd_y//2, wcs, origin=1)
        ra = frameCentre.ra.degree
        dec = frameCentre.dec.degree

        if '_' in self.object:
            asteroid = self.object.replace('_', ' ')
        else:
            asteroid = self.object

        RA_horizons, DEC_horizons, airmass = self.get_horizons_info(jd)

        horizons_coord = SkyCoord(ra=RA_horizons, dec=DEC_horizons, unit='deg', frame='icrs')

        catalogue = ascii.read(catalogueName, format='sextractor')
        catalogue = catalogue[
            (catalogue['FWHM_IMAGE'] > 1) &
            (catalogue['FLAGS'] == 0) &
            (catalogue['ISOAREA_IMAGE'] > 6) &
            (catalogue['FLUX_MAX'] < 57500)
        ]

        # Find the position angle of each source and 
        # filter the catalogue to only show non-trailed sources.
        # This helps to separate the asteroid from nearby trailed stars in the catalogue.
        # High elongation bound allows for poor tracking or wind shake.

        elongation = catalogue['ELONGATION']

#         catalogue = catalogue[
#             (elongation >= 1) &
#             (elongation < 1.75)
#         ]

        pdCat = catalogue.to_pandas()
        frameArr = []

        for index, row in pdCat.iterrows():
            # Read in RA & DEC of object in catalogue and set astrometric precision
            # at 5 arcseconds
            ra = row['ALPHA_J2000']
            dec = row['DELTA_J2000']

            catalogueCoord = SkyCoord(ra=ra, dec=dec, unit='deg')
            separation = horizons_coord.separation(catalogueCoord).arcsecond
            deltaRA = horizons_coord.ra.arcsecond - catalogueCoord.ra.arcsecond
            deltaDEC = horizons_coord.dec.arcsecond - catalogueCoord.dec.arcsecond

            flux = row['FLUX_AUTO']
            mag = -2.5 * np.log10(flux/exptime)
            mag_err = row['MAGERR_AUTO']
            
            A_IMAGE = row['A_IMAGE']
            B_IMAGE = row['B_IMAGE']
            KRON_RADIUS = row['KRON_RADIUS']
            background = row['BACKGROUND']
            
            kron_ellipse_area = np.pi * A_IMAGE * KRON_RADIUS * B_IMAGE * KRON_RADIUS
            
            total_background = background * kron_ellipse_area
            
            SNR = flux/np.sqrt(flux + total_background)
            
            ellipticity = row['ELLIPTICITY']

            frameArr.append([jd, mag, mag_err, separation, airmass, deltaRA, deltaDEC, exptime, SNR, ellipticity])

        df = pd.DataFrame(frameArr, columns=['JD', 'MAG_AUTO', 'MAGERR_AUTO', 'Separation', 'AIRMASS','deltaRA', 'deltaDEC', 'EXPTIME', 'SNR', 'ELLIPTICITY'])
        # Return the catalogue object that is closest to the predicted position of the asteroid
        return df[df['Separation'] == df['Separation'].min()].values[0]


    def get_colour(self, rMags, vMags):
        rMags = rMags.astype('float')
        vMags = vMags.astype('float')
        minJd = rMags.JD.min()
        maxJD = rMags.JD.values[-1]
        
        # Create a 1d interpolation spline of the R values
        spline = interp1d(rMags.JD.values, rMags.RELMAG.values)
        
        # Get a list of V mag julian dates than fall within this spline
        vJDs = [jd for jd in vMags.JD.values if minJd <= jd <= maxJD]
        
        plt.figure(1, (14, 8))
        plt.scatter(rMags.JD.values - minJd, rMags.RELMAG.values, c='r', label='R-filter')
        plt.scatter(vMags.JD.values - minJd, vMags.RELMAG.values, c='g', label='V-filter')
        plt.plot(rMags.JD.values - minJd, spline(rMags.JD.values), c='r')

        rInterp = []
        rInterpErr = []
        for i, jd in enumerate(vJDs):
            # Find the R mag at the JD for this V frame
            y = spline(jd)
            rInterp.append(y)
            
            p1 = patches.FancyArrowPatch((jd - minJd, vMags.RELMAG.values[i]), (jd - minJd, y), arrowstyle='<->', mutation_scale=20)
            plt.gca().add_patch(p1)
            colour = vMags.RELMAG.values[i] - y
            plt.annotate(f'{colour:.2f}', (jd - minJd, np.mean([vMags.RELMAG.values[i], y])), xytext=(1, 0), textcoords='offset pixels', fontsize=10)
            # Create an uncertainty for this point using the 
            # average of the uncertainties of the bounding points
            beforeErr = rMags[
                rMags.JD == rMags[rMags.JD < jd]['JD'].max()
            ]['RELMAG_ERR'].values[0]
            afterErr = rMags[
                rMags.JD == rMags[rMags.JD > jd]['JD'].min()
            ]['RELMAG_ERR'].values[0]
            
            rInterpErr.append(np.sqrt(np.square(beforeErr) + np.square(afterErr))/2)

        rMagnitudes = pd.DataFrame(
            list(zip(vJDs, rInterp, rInterpErr)),
            columns=["JD", "RELMAG_R", "RELMAG_ERR_R"]
        )
        plt.scatter(rMagnitudes.JD.values - minJd, rMagnitudes.RELMAG_R, c='orange', label='Interpolated R-filter')
        plt.xlabel(f'Julian Date - {minJd}')
        plt.ylabel('Relative Magnitude')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(self.baseFolder + f'Interp.pdf', bbox_inches='tight')
        plt.close()
        # Create dataframe of V & R values at specific JDs
        mergedDf = rMagnitudes.merge(vMags, how='inner', on='JD')
        
        # Calculate the colour of each pair and corresponding uncertainty
        mergedDf['colour'] = mergedDf['RELMAG'] - mergedDf['RELMAG_R']
        mergedDf['colour_err'] = np.sqrt(
            np.square(mergedDf['RELMAG_ERR']) + np.square(mergedDf['RELMAG_ERR_R'])
        )
        
        # Calculate the sigma-clipped mean, median, and std. dev of the colour
        mean, median, std = sigma_clipped_stats(mergedDf['colour'].astype(float),
                                                cenfunc='median')

        unc = std # /np.sqrt(len(mergedDf.colour))

        # Plot the colour as a function of JD with 1-sigma bounds
        plt.rcParams.update({'font.size': 20})
        plt.figure(1, (14, 8))
        mergedDf = mergedDf.astype('float64')
        plt.errorbar(
            x=mergedDf.JD, y=mergedDf.colour,
            yerr=mergedDf.colour_err, fmt='o', c='k'
        )
        plt.xlabel('Julian Date')
        plt.ylabel('$V_{inst.} - R_{inst.}$')
        plt.gca().invert_yaxis()
        plt.gca().axhline(mean, linestyle='--', color='k')
        plt.gca().axhspan(
            mean-unc, mean+unc
        )
        plt.savefig(self.baseFolder + f'InstColour.pdf', bbox_inches='tight')
        plt.close()
        
        return mean, unc
        
        
    def get_ps1_stars_all(self, filt):

        frames = self.filter_frame_lists[filt]
        idx = self.best_filter_frames[filt]

        ps1path = f"{frames[idx].strip('.fits')}.csv"
        
        # This assumes that all the images have the same exposure time
        # as the first image
        with fits.open(frames[idx]) as image:
            image[0].verify('silentfix')
            header = image[0].header
            exptime = header['EXPTIME']

        # Extracts the World Coordinate System info from the header
        # (Image should be plate solved from the pipeline
        wcs = WCS(header)

        # Determines the size of the image in pixels
        ccd_x = header['IMAGEW']
        ccd_y = header['IMAGEH']

        # Converts the centre pixel to RA & DEC (origin=1 required for FITS
        # images)
        frameCentre = utils.pixel_to_skycoord(ccd_x//2, ccd_y//2, wcs,
                                            origin=1)
        ra = frameCentre.ra.degree
        dec = frameCentre.dec.degree

        # Sets the search radius to 16 arcminutes for pirate
        radius = self.FoV/60.0

        # Require stars with more than one detection in the r & g filters
        constraints = {'nDetections.gt': 1, 'ng.gt': 1, 'nr.gt': 1}

        # Strip blanks and weed out blank and commented-out values
        columns = """raMean,decMean,nDetections,ng,nr,
            gMeanPSFMag,rMeanPSFMag,
            gMeanKronMag,rMeanKronMag,
            gMeanPSFMagErr,rMeanPSFMagErr,
            gMeanKronMagErr,rMeanKronMagErr""".split(',')
        columns = [x.strip() for x in columns]
        columns = [x for x in columns if x and not x.startswith('#')]

        # Find stars matching criteria in the PanStarrs catalogue
        results = ps1cone(ra, dec, radius, release='dr2', columns=columns,
                        verbose=True, **constraints)

        # Write catalogue output to file
        f = open(ps1path, 'w')
        f.write(results)
        f.close()
        
        i = 0
        ZParr = np.zeros((len(frames), 2))
        for frame in frames:
            print(frame)
            # Define naming convention for files using base name of image
            baseName = frame.strip('.fits')
            # Define catalogue name from pipeline
            catalogueName = (
                self.baseFolder + 'Catalogues/' + baseName.split('/')[-1] + '.cat'
            )

            # If PanStarrs output exists for this field
            # Read the csv into an Astropy ascii table
            tab = ascii.read(ps1path, format='csv', delimiter=',')

            # Improve the format
            for filter in 'gr':

                col = filter+'MeanPSFMag'
                try:
                    tab[col].format = ".4f"
                    tab = tab[tab[col] != -999.0]
                except KeyError:
                    print("{} not found".format(col))

                col = filter+'MeanPSFMagErr'
                try:
                    tab[col].format = ".4f"
                    tab = tab[tab[col] != -999.0]
                    tab = tab[tab[col] != 0.0]
                except KeyError:
                    print("{} not found".format(col))

                col = filter+'MeanKronMag'
                try:
                    tab[col].format = ".4f"
                    tab = tab[tab[col] != -999.0]
                except KeyError:
                    print("{} not found".format(col))

                col = filter+'MeanKronMagErr'
                try:
                    tab[col].format = ".4f"
                    tab = tab[tab[col] != -999.0]
                    tab = tab[tab[col] != 0.0]
                except KeyError:
                    print("{} not found".format(col))

            # Filter output from PanStarrs catalogue further as follows:
            #
            #   Number of detections in each filter greater than 2
            #   Difference between PSF and Kron magnitudes less than 0.05
            #   Errors on the PSF magnitude less than 0.008 mag (currently just r)
            #   No colours greater than or equal to 1.5 mag
            #   No stars with r magnitude less than 16 mag
            #
            # Reason for PSF and Kron magnitude difference limit is to remove extended
            # sources. If the Kron magnitude is much different from the PSF magnitude
            # then the PSF is likely non-gaussian (poor PSF fitting) and hence is
            # likely an extended source.

            tabb = tab[(tab['nr'] > 2) &
                    (tab['ng'] > 2) &
                    (tab['rMeanPSFMag'] - tab['rMeanKronMag'] < 0.05) &
                    (tab['rMeanPSFMagErr'] < 0.008) &
                    (tab['gMeanPSFMag'] - tab['rMeanPSFMag'] < 1.5) &
                    (tab['rMeanPSFMag'] < 17)]

            # Read in the source extractor catalogue and filter it as follows:
            # Objects must have a FWHM greater than one pixel (removes artifacts)
            # Objects can't have flags indicating photometry errors
            # Area of object in the image must be greater than six pixels

            catalogue = ascii.read(catalogueName, format='sextractor')
            catalogue = catalogue[(catalogue['FWHM_IMAGE'] > 1) &
                                (catalogue['FLAGS'] == 0) &
                                (catalogue['ISOAREA_IMAGE'] > 6) &
                                (catalogue['FLUX_MAX'] < 57500) &
                                (catalogue['FLUX_MAX'] > 500)]

            # Find the position angle of each source and determine the sigma-clipped
            # mean position angle. With trailed images all of the stars should be
            # trailed in the same direction. We therefore ignore all objects with
            # position angles more than 3-sigma away from the mean.

            theta = catalogue['THETA_IMAGE']
            mean, median, std = sigma_clipped_stats(theta, cenfunc='mean')
            catalogue = catalogue[abs(catalogue['THETA_IMAGE']-mean) < 3*std]

            # Convert Astropy ascii tables to pandas for easier manipulation and loop
            # over all rows in the dataframe representing the PanStarrs catalogue
            pdCatalogue = catalogue.to_pandas()
            pdTabb = tabb.to_pandas()
            # Initialise empty array to contain information about matched stars
            arr = []

            for index, row in pdTabb.iterrows():
                # Read in RA & DEC of object in catalogue and set astrometric precision
                # at 2 arcseconds
                ra = row['raMean']
                dec = row['decMean']
                arcsec = 5.0
                precision = arcsec/3600.0

                # Return entries in the sextractor catalogue within the astrometric
                # precision of the selected star from the PanStarrs catalogue
                catalogueEntryMatch = pdCatalogue[
                    (abs(pdCatalogue['ALPHA_J2000'] - ra) < precision) &
                    (abs(pdCatalogue['DELTA_J2000'] - dec) < precision)
                ]

                # If a unique star is found within the astrometric precision then it is
                # taken to be a match within the catalogues
                if len(catalogueEntryMatch) == 1:

                    # Extract the relevent measured paramters from the sextractor
                    # catalogue for the matched star
                    flux = catalogueEntryMatch['FLUX_AUTO'].values[0]
                    R_frame = -2.5 * np.log10(flux/exptime)
                    R_frame_err = catalogueEntryMatch['MAGERR_AUTO'].values[0]
                    ALPHA_J2000 = catalogueEntryMatch['ALPHA_J2000'].values[0]
                    DELTA_J2000 = catalogueEntryMatch['DELTA_J2000'].values[0]

                    # Extract the relevant parameters from the PanStarrs catalogue for
                    # the matched star
                    g_P1 = row['gMeanPSFMag']
                    r_P1 = row['rMeanPSFMag']
                    g_P1_err = row['gMeanPSFMagErr']
                    r_P1_err = row['rMeanPSFMagErr']

                    # Calculate the PanStarrs colour and it's corresponding error by
                    # adding the individual errors in quadrature
                    colour = g_P1 - r_P1
                    colourErr = np.sqrt(np.square(g_P1_err) + np.square(r_P1_err))

                    # Calculate the difference between instrumental and PanStarrs
                    # magnitude and its corresponding uncertainty by adding the individual
                    # errors in quadrature
                    diff = R_frame - r_P1
                    diffErr = np.sqrt(np.square(R_frame_err) + np.square(r_P1_err))

                    # Append the required information to the array of matched stars
                    arr.append([str(ra), str(dec), str(ALPHA_J2000), str(DELTA_J2000),
                                str(R_frame), str(R_frame_err), str(g_P1),
                                str(g_P1_err), str(r_P1), str(r_P1_err), str(colour),
                                str(colourErr), str(diff), str(diffErr)])

            # Convert array to a dataframe for ease of manipulation
            df = pd.DataFrame(arr, columns=['RA', 'DEC', 'ALPHA_J2000', 'DELTA_J2000',
                                            'R_frame', 'R_frame_err', 'g_P1',
                                            'g_P1_err', 'r_P1', 'r_P1_err', 'colour',
                                            'colourErr', 'diff', 'diffErr'])
            df.to_csv(self.baseFolder + f'{self.object}-{self.date}-PS1-matched-stars-{i}.csv', index=False)
            
            mean, median, std = sigma_clipped_stats(df['diff'].astype('float64'),
                                            cenfunc='median')
            sigma = 2.5
            df = df[abs(df['diff'].astype('float64') - median) < sigma * std]

            x = df['colour'].astype('float64')
            y = df['diff'].astype('float64')
            x_err = df['colourErr'].astype('float64')
            y_err = df['diffErr'].astype('float64')
            x_fit = np.linspace(min(x), max(x), 100)

            def func(x, c):
                return self.CT*x + c


            popt_init, pcov_init = curve_fit(func, x, y, sigma=y_err)
            perr_init = np.sqrt(np.diag(pcov_init))


            def func1(p, x):
                c = p
                return -0.0353*x + c


            def func2(m, c, x):
                return m*x + c


            # Model object
            lin_model = odr.Model(func1)


            # Create a RealData object
            data = odr.RealData(x, y, sx=x_err, sy=y_err)
            # Set up ODR with the model and data.
            odr1 = odr.ODR(data, lin_model, beta0=popt_init)

            # Run the regression.
            out = odr1.run()

            # ZP & ZPerr
            popt = out.beta
            perr = out.sd_beta

            nstd = 3.
            popt_up = popt + nstd * perr
            popt_dw = popt - nstd * perr
            
            CT_up = self.CT + nstd * self.CTerr
            CT_dw = self.CT - nstd * self.CTerr

            fit = func1(popt, x_fit)
            fit_up = func2(CT_up, popt_up, x_fit)
            fit_dw = func2(CT_dw, popt_dw, x_fit)

            # plot colour graph

            plt.rcParams.update({'font.size': 16})
            plt.rcParams.update({'mathtext.default':  'regular'})
            plt.figure(1, (12, 8))
            plt.errorbar(x, y, yerr=y_err, xerr=x_err, ecolor='k', fmt='o',
                        markerfacecolor='none', label='Matched Stars')
            plt.xlabel('PanStarrs Colour ($g_{P1} - r_{P1}$)')
            plt.ylabel('$R_{frame} - r_{P1}$')
            plt.fill_between(x_fit, fit_up, fit_dw, alpha=.25,
                            label=r'3-$\sigma$ interval')
            plt.plot(x_fit, fit, 'r', lw=2, label='ODR fit w/ xerr & yerr')
            plt.gca().text(0.01, 0.95, fr'$CT = {self.CT} \pm {self.CTerr}$',
                        verticalalignment='top', horizontalalignment='left',
                        transform=plt.gca().transAxes,
                        color='green', fontsize=15)
            plt.gca().text(0.01, 0.90,
                        r'$ZP = {:.4f} \pm {:.4f}$'.format(popt[0], perr[0]),
                        verticalalignment='top', horizontalalignment='left',
                        transform=plt.gca().transAxes,
                        color='green', fontsize=15)
            plt.legend(loc='best')
            plt.savefig(self.baseFolder + f'colourPlot-{i}.pdf', bbox_inches='tight')
            plt.close()
            ZParr[i] = [popt[0], perr[0]]
            i += 1
            
        return ZParr


    def get_geometry_horizons(self, julian_date):

        horizonsString = (
            "https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1" +
            f"&COMMAND='{self.object}%3B'" +
            "&CENTER='coord@399'" +
            "&COORD_TYPE='GEODETIC'" +
            "&SITE_COORD='" +
            f"{self.location['lon']},{self.location['lat']},{self.location['elevation']}'" +
            "&MAKE_EPHEM='YES'" +
            "&TABLE_TYPE='OBSERVER'" +
            f"&TLIST='{julian_date}'" +
            "&CAL_FORMAT='CAL'" +
            "&TIME_DIGITS='SECONDS'" +
            "&ANG_FORMAT='DEG'" +
            "&OUT_UNITS='KM-S'" +
            "&RANGE_UNITS='AU'" +
            "&APPARENT='AIRLESS'" +
            "&SUPPRESS_RANGE_RATE='NO'" +
            "&SKIP_DAYLT='NO'" +
            "&EXTRA_PREC='YES'" +
            "&R_T_S_ONLY='NO'" +
            "&REF_SYSTEM='J2000'" +
            "&CSV_FORMAT='YES'" +
            "&OBJ_DATA='NO'" +
            "&QUANTITIES='2,3,19,20,21,24,39'"
        )

        response = requests.get(horizonsString)
        data = response.text
        start = "$$SOE"
        end = "$$EOE"
        ephem = data[data.find(start)+len(start):data.rfind(end)]
        ephem = ephem.replace(" ", "").replace("\n", "").split(',')
        
        RA_rate = float(ephem[5]) / (np.cos(float(ephem[3])) * 60)
        DEC_rate = float(ephem[6]) / 60

        rate_of_motion = np.sqrt(np.square(RA_rate) + np.square(DEC_rate))

        heliocentric_dist = float(ephem[7])
        geocentric_dist = float(ephem[9])
        LTT = float(ephem[11])/(60*24)
        phase_angle = float(ephem[12])
        range_unc = float(ephem[13]) / 1.49597871e8

        return heliocentric_dist, geocentric_dist, LTT, phase_angle, range_unc, rate_of_motion
        
        
    def fit_pair_transformation_intercept(self):

        rFrames = self.filter_frame_lists['R']
        refFrameR = self.best_filter_frames['R']
        rFrame = rFrames[refFrameR]

        vFrames = self.filter_frame_lists['V']
        refFrameV = self.best_filter_frames['V']
        vFrame = vFrames[refFrameV]

        # Read the Pan-Starrs catalogue for this field
        ps1path = rFrame.strip('.fits') + '.csv'
        tab = ascii.read(ps1path, format='csv', delimiter=',')
        
        # Defines the sextractor catalogue path for the V frame
        baseNameV = vFrame.strip('.fits')
        catalogueNameV = self.baseFolder + 'Catalogues/' + \
            baseNameV.split('/')[-1] + '.cat'

        # Defines the sextractor catalogue path for the R frame
        baseNameR = rFrame.strip('.fits')
        catalogueNameR = self.baseFolder + 'Catalogues/' + \
            baseNameR.split('/')[-1] + '.cat'

        # Open each image and read in values from the header
        imageV = fits.open(vFrame)
        imageR = fits.open(rFrame)
        imageV[0].verify('silentfix')
        imageR[0].verify('silentfix')
        headerV = imageV[0].header
        headerR = imageR[0].header
        exptimeV = headerV['EXPTIME']
        exptimeR = headerR['EXPTIME']
        imageV.close()
        imageR.close()

        # Improve the format
        for filter in 'gr':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab = tab[tab[col] != -999.0]
            except KeyError:
                print("{} not found".format(col))

            col = filter+'MeanPSFMagErr'
            try:
                tab[col].format = ".4f"
                tab = tab[tab[col] != -999.0]
                tab = tab[tab[col] != 0.0]
            except KeyError:
                print("{} not found".format(col))

            col = filter+'MeanKronMag'
            try:
                tab[col].format = ".4f"
                tab = tab[tab[col] != -999.0]
            except KeyError:
                print("{} not found".format(col))

            col = filter+'MeanKronMagErr'
            try:
                tab[col].format = ".4f"
                tab = tab[tab[col] != -999.0]
                tab = tab[tab[col] != 0.0]
            except KeyError:
                print("{} not found".format(col))

        tabb = tab[(tab['nr'] > 2) &
                (tab['ng'] > 2) &
                (tab['rMeanPSFMag'] - tab['rMeanKronMag'] < 0.05) &
                (tab['rMeanPSFMagErr'] < 0.008) &
                (tab['gMeanPSFMag'] - tab['rMeanPSFMag'] < 1.5) &
                (tab['rMeanPSFMag'] < 17)]

        catalogueV = ascii.read(catalogueNameV, format='sextractor')
        catalogueV = catalogueV[(catalogueV['FWHM_IMAGE'] > 1) &
                                (catalogueV['FLAGS'] == 0) &
                                (catalogueV['ISOAREA_IMAGE'] > 6) &
                                (catalogueV['FLUX_MAX'] < 57500) &
                                (catalogueV['FLUX_MAX'] > 200)]

        catalogueR = ascii.read(catalogueNameR, format='sextractor')
        catalogueR = catalogueR[(catalogueR['FWHM_IMAGE'] > 1) &
                                (catalogueR['FLAGS'] == 0) &
                                (catalogueR['ISOAREA_IMAGE'] > 6) &
                                (catalogueR['FLUX_MAX'] < 57500) &
                                (catalogueR['FLUX_MAX'] > 200)]

        theta = catalogueV['THETA_IMAGE']
        mean, median, std = sigma_clipped_stats(theta, cenfunc='mean')
        catalogueV = catalogueV[abs(catalogueV['THETA_IMAGE']-mean) < 3*std]

        theta = catalogueR['THETA_IMAGE']
        mean, median, std = sigma_clipped_stats(theta, cenfunc='mean')
        catalogueR = catalogueR[abs(catalogueR['THETA_IMAGE']-mean) < 3*std]

        pdCatalogueV = catalogueV.to_pandas()
        pdCatalogueR = catalogueR.to_pandas()
        pdTabb = tabb.to_pandas()

        # Initialise empty array to contain information about matched stars
        arr = []

        for index, row in pdTabb.iterrows():
            ra = row['raMean']
            dec = row['decMean']
            arcsec = 5.0
            precision = arcsec/3600.0

            # Return entries in the sextractor catalogue within the astrometric
            # precision of the selected star from the PanStarrs catalogue
            catalogueEntryMatchV = pdCatalogueV[
                (abs(pdCatalogueV['ALPHA_J2000'] - ra) < precision) &
                (abs(pdCatalogueV['DELTA_J2000'] - dec) < precision)
            ]

            catalogueEntryMatchR = pdCatalogueR[
                (abs(pdCatalogueR['ALPHA_J2000'] - ra) < precision) &
                (abs(pdCatalogueR['DELTA_J2000'] - dec) < precision)
            ]

            # print(f"V: {len(catalogueEntryMatchV)}")
            # print(f"R: {len(catalogueEntryMatchR)}")
            if (
                len(catalogueEntryMatchV) == 1 and
                len(catalogueEntryMatchR) == 1
            ):

                # Extract the relevant measured paramters from the sextractor
                # catalogue for the matched star
                fluxR = catalogueEntryMatchR['FLUX_AUTO'].values[0]
                fluxV = catalogueEntryMatchV['FLUX_AUTO'].values[0]
                R_frame = -2.5 * np.log10(fluxR/exptimeR)
                V_frame = -2.5 * np.log10(fluxV/exptimeV)
                R_frame_err = catalogueEntryMatchR['MAGERR_AUTO'].values[0]
                V_frame_err = catalogueEntryMatchV['MAGERR_AUTO'].values[0]

                g_P1 = row['gMeanPSFMag']
                r_P1 = row['rMeanPSFMag']
                g_P1_err = row['gMeanPSFMagErr']
                r_P1_err = row['rMeanPSFMagErr']

                colour = g_P1 - r_P1
                colourErr = np.sqrt(np.square(g_P1_err) + np.square(r_P1_err))

                instColour = V_frame - R_frame
                instColourErr = np.sqrt(np.square(R_frame_err) +
                                        np.square(V_frame_err))

                arr.append([ra, dec, colour, colourErr,
                            instColour, instColourErr])

        # Convert array to a dataframe for ease of manipulation
        df = pd.DataFrame(arr, columns=['RA', 'DEC', 'colour', 'colourErr',
                                        'instColour', 'instColourErr'])
        df.fillna(value=0, inplace=True)
        if len(df) == 0:
            raise Exception("No data to calculate colour.")
        
        df.to_csv(self.baseFolder + 'colour_transformation_data.csv')

        x = df.instColour
        x_err = df.instColourErr
        y = df.colour
        y_err = df.colourErr
        
        x_fit = np.linspace(min(x), max(x), 50)
        
        # initialize a linear fitter
        fit = fitting.LevMarLSQFitter()

        # initialize the outlier removal fitter
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=5, sigma=3.0)

        # initialize a linear model
        # old slope = 1.5878
        # old slope (2020/10/29) = 1.3926 +- 0.0026
        # PIRATE Mk.4 slope (2021/07/26)
        line_init = models.Linear1D(slope=self.CTG, fixed={'slope': True})

        # fit the data with the fitter
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            fitted_line, mask = or_fit(line_init, x, y, weights=1.0/x_err)

        filtered_y = y[~mask]
        filtered_yerr = y_err[~mask]
        filtered_x = x[~mask]
        filtered_xerr = x_err[~mask]

        clipped_x = x[mask]
        clipped_xerr = x_err[mask]
        clipped_y = y[mask]
        clipped_yerr = y_err[mask]

        pcov = or_fit.fitter.fit_info['param_cov']
        perr = np.sqrt(np.diag(pcov))
        warnings.simplefilter('error', UserWarning)
        c = fitted_line.intercept.value
        c_err = perr[0]

        plt.rcParams.update({'font.size': 16})
        plt.figure(1, (12, 10))
        plt.errorbar(clipped_x, clipped_y, yerr=clipped_yerr, xerr=clipped_xerr, fmt="ko", elinewidth=1, fillstyle="none", label="Clipped Data")
        plt.errorbar(filtered_x, filtered_y, yerr=filtered_yerr, xerr=filtered_xerr, fmt="ko", elinewidth=1, label="Fitted Data")
        plt.plot(x_fit, fitted_line(x_fit), 'k--', label='Fitted Model')
        plt.gca().text(0.01, 0.95,
                    r'$c = {:.4f} \pm {:.4f}$'.format(c, c_err),
                    verticalalignment='top', horizontalalignment='left',
                    transform=plt.gca().transAxes, color='r',
                    fontsize=15)
        plt.xlabel('$V_{inst.} - R_{inst.}$')
        plt.ylabel('$g_{P1} - r_{P1}$')
        plt.legend(loc='lower right')
        plt.savefig(self.baseFolder + f'ColourTransformation.pdf', bbox_inches='tight')
        plt.close()

        return c, c_err