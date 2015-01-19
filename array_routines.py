# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 07:40:04 2015

@author: davcra


Functions for array analsis of microseism locations.



"""

import scipy as sp
import numpy as np
import scipy.ndimage
from obspy.signal.array_analysis import *
from obspy.core import UTCDateTime, read, AttribDict
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import mlab

import mpl_toolkits.basemap.pyproj as pyproj
import time
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator
from obspy.signal.util import az2baz2az
from obspy.signal.util import prevpow2
from obspy.signal.spectral_estimation import fft_taper, psd



matplotlib.rcParams.update({'font.size': 16})

# build colormap as done in paper by mcnamara
CDICT = {'red': ((0.0, 1.0, 1.0),
                 (0.05, 1.0, 1.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.05, 0.0, 0.0),
                   (0.2, 0.0, 0.0),
                   (0.4, 1.0, 1.0),
                   (0.6, 1.0, 1.0),
                   (0.8, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.05, 1.0, 1.0),
                  (0.2, 1.0, 1.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
colormap = LinearSegmentedColormap('mcnamara', CDICT, 1024)


def insert_coordinates(stream, coordFile):
    """
    Helper function to write coordinate details into an ObsPy Stream object headers from a 
    text file for array analysis.

    :type stream: obspy stream object.
    :param stream: obspy stream object containing data for each array station.
    :type coordFile: string.
    :param coordFile: text file with headers trace.id, longitude, latitude and 
        elevation.
    :return: Stream object, where each trace.stats contains an obspy.core.util.AttribDict 
        with 'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x', 'y', 
        'elevation' (in km) items/attributes.
    """
    coordinates = open(coordFile, 'r')
    for line in coordinates:
        c = (line.strip('\n').split('\t'))
        for tr in stream:
            if tr.id == c[0]:
                tr.stats.coordinates = AttribDict({'latitude': c[2],
                                                       'elevation': c[3],
                                                       'longitude': c[1]})
    return stream
    
    
    
    
def array_processing(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y,
                     sl_s, semb_thres, vel_thres, frqlow, frqhigh, stime,
                     etime, prewhiten, verbose=False, coordsys='lonlat',
                     timestamp='mlabday', method=0, diag=True, diag_fact=0.01,
                     store=None, get_lims=True, plot=None, plotspec=False):
                         
    """
    Method for Seismic-Array-Beamforming/FK-Analysis/Capon

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type semb_thres: Float
    :param semb_thres: Threshold for semblance
    :type vel_thres: Float
    :param vel_thres: Threshold for velocity
    :type frqlow: Float
    :param frqlow: lower frequency for fk/capon
    :type frqhigh: Float
    :param frqhigh: higher frequency for fk/capon
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type prewhiten: int
    :param prewhiten: Do prewhitening, values: 1 or 0
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type method: int
    :param method: the method to use 0 == bf, 1 == capon
    :type diag: bool
    :param diag: If True use diagonal loading for capon spectrum (recommended).
    :type diag_fact: float
    :param diag_fact: Gives some control on weights applied to diagonal element 
        of covariance matrix. Try between 0.01 and 0.1. Too high and peak in 
        slowness spectrum becomes broad, too low and effect of weights is lost.
        Experiment using plot function.
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :type plot: function
    :param plot: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.        
    :type plotspec: bool    
    :param plotspec: if true axis' for plot are revesed to show baz instead 
                    of az.
        
        
        
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness
    """
    res=[]
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift(geometry, sll_x, sll_y,
                                     sl_s, grdpts_x, grdpts_y)
    # offset of arrays
    spoint, _epoint = get_spoint(stream, stime, etime)
    #
    # loop with a sliding window over the dat trace array and apply bbfk
    #
    nstat = len(stream)
    fs = stream[0].stats.sampling_rate
    nsamp = int(win_len * fs)
    nstep = int(nsamp * win_frac)

    # generate plan for rfftr
    nfft = nextpow2(nsamp)
    deltaf = fs / float(nfft)
    nlow = int(frqlow / float(deltaf) + 0.5)
    nhigh = int(frqhigh / float(deltaf) + 0.5)
    nlow = max(1, nlow)  # avoid using the offset
    nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency
       
    
    # to spead up the routine a bit we estimate all steering vectors in advance
    steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16') 
    clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                         deltaf, time_shift_table, steer)

    R = np.empty((nf, nstat, nstat), dtype='c16')    
    ft = np.empty((nstat, nf), dtype='c16')           
    
    newstart = stime
    tap = np.hanning(nsamp) #cosTaper(nsamp, p=0.22)  # 0.22 matches 0.2 of historical C bbfk.c
    offset = 0
    relpow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    RPOW = np.empty((grdpts_x, grdpts_y), dtype='f8')
    

    while eotr:
        try:
            for i, tr in enumerate(stream):
                dat = tr.data[spoint[i] + offset:
                              spoint[i] + offset + nsamp]
                dat = (dat - dat.mean()) * tap
                ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]
        except IndexError:
            break
        
        ft = np.require(ft, 'c16', ['C_CONTIGUOUS'])
        relpow_map.fill(0.)
        abspow_map.fill(0.)
        RPOW.fill(0.)
        # computing the covariances of the signal at different receivers
        dpow = 0.
        for i in xrange(nstat):
            for j in xrange(i, nstat):
                R[:, i, j] = ft[i, :] * ft[j, :].conj()
#                if method == 1:
#                    R[:, i, j] /= np.abs(R[:, i, j].sum()**2)   # why?
                if i != j:
                    R[:, j, i] = R[:, i, j].conjugate()
                else:
                    dpow +=  np.abs(R[:, i, j].sum())
        dpow *= nstat
        
        # Equation for conv beamforming
        # P(f) = e.H R(f) e
        # P(f) = steer^H dot R dot steer
        # where e is the covariance (steering) matrix and 
        # R is the cross-spectral matrix.        
        
        if method == 1:
            # Equation for Capons beamformer
            # P(f) = 1/(e.H R(f)^-1 e)
            # P(f) = 1 / (steer^H dot R dot steer)
            # R(f)^-1 can be nonsingular (i.e. det=0) so diagonal loading is 
            # applied (Capon1969).
            if diag == True:

                # translated from gal code
                I = np.identity(nstat)
                for n in xrange(nf):

                    # calculate weights (capon1969)
                    # Vectorised version of nice DOA codes by M. Gal available 
                    # at https://github.com/mgalcode
                    weights = I*R[n, :, :].real.trace()/(nsamp)*diag_fact
                    R[n, :, :].real += weights
                    w = np.abs(R[n].diagonal())**-0.5  
                    wmean = np.sum(1./w**2)
                    wmean /= float(nstat)*(nhigh)                    
                    w_prod = np.outer(w, w.T)
                    R[n] *= w_prod
            
            for n in xrange(nf):
                R[n, :, :] = np.linalg.pinv(R[n, :, :], rcond=1e-6)
            
        
        errcode = clibsignal.generalizedBeamformer(
            relpow_map, abspow_map, steer, R, nsamp, nstat, prewhiten,
            grdpts_x, grdpts_y, nfft, nf, dpow, method) 
        if errcode != 0:
            msg = 'generalizedBeamforming exited with error %d'
            raise Exception(msg % errcode)
            
            
        
        #plot rpow
        if plot:
            param_estimation(relpow_map, 0.8, UTCDateTime(newstart.timestamp), 
                             sll_x, sll_y, sl_s)
            net = stream[0].stats.network
            cfreq = (frqhigh+frqlow) / 2
            flim = frqhigh - cfreq
            plot(relpow_map, net, UTCDateTime(newstart.timestamp), cfreq, flim,
                 sll_x, sll_y, slm_x, slm_y, sl_s)
        
            
        RPOW += relpow_map
            
        ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
        relpow, abspow = relpow_map[ix, iy], abspow_map[ix, iy]
        if store is not None:
            store(relpow_map, abspow_map, offset)
        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180
        
        
        if relpow > semb_thres and 1. / slow > vel_thres:
            res.append(np.array([newstart.timestamp, relpow, abspow, baz,
                                 slow]))
            if verbose:
                print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
        if (newstart + (nsamp + nstep) / fs) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / fs
        
        
        
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
        # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719163
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
        
    
    return np.array(res), RPOW

    
    
    
def dir_spec(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y,
                  sl_s, semb_thres, vel_thres, frqlow, frqhigh, nfft,
                  stime, etime, prewhiten, verbose=False, 
                  coordsys='lonlat', timestamp='mlabday', method=0, 
                  diag=True, diag_fact=0.01, store=None):
    
    """
   **** Not Working Yet ****    
    
    Method for Seismic-Array-Beamforming/FK-Analysis/Capon

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type semb_thres: Float
    :param semb_thres: Threshold for semblance
    :type vel_thres: Float
    :param vel_thres: Threshold for velocity
    :type frqlow: Float
    :param frqlow: lower frequency for fk/capon
    :type frqhigh: Float
    :param frqhigh: higher frequency for fk/capon
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type prewhiten: int
    :param prewhiten: Do prewhitening, values: 1 or 0
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type method: int
    :param method: the method to use 0 == bf, 1 == capon
    :type diag: bool
    :param diag: If True use diagonal loading for capon spectrum (recommended).
    :type diag_fact: float
    :param diag_fact: Gives some control on weights applied to diagonal element 
        of covariance matrix. Try between 0.01 and 0.1. Too high and peak in 
        slowness spectrum becomes broad, too low and effect of weights is lost.
        Experiment using plot function.
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :type plot: function
    :param plot: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.        
    :type plotbaz: bool    
    :param plotbaz: if true axis' for plot are revesed to show baz instead 
                    of az.
    
    
    
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness
    """
    res = []
    fbaz = []
    fslow = []        
    
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift(geometry, sll_x, sll_y,
                                     sl_s, grdpts_x, grdpts_y)
    # offset of arrays
    spoint, _epoint = get_spoint(stream, stime, etime)
    #
    # loop with a sliding window over the dat trace array and apply bbfk
    #
    nstat = len(stream)
    fs = stream[0].stats.sampling_rate
    nsamp = int(win_len * fs)
    nstep = int(nsamp * win_frac)
        
    
    # generate plan for rfftr
    
    
    # Loop over fbins from here    
    
    #nfft = prevpow2(nsamp)
    deltaf = fs / float(nfft)
    nlow = int(frqlow / float(deltaf) + 0.5)
    nhigh = int(frqhigh / float(deltaf) + 0.5)
    nlow = max(1, nlow)  # avoid using the offset
    nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency
    freq_bins = np.fft.fftfreq(nfft, d=1)[nlow:nlow + nf]
     

    # to spead up the routine a bit we estimate all steering vectors in advance
    steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16') 
    clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                         deltaf, time_shift_table, steer)

    R = np.empty((nf, nstat, nstat), dtype='c16')    # cross-spectral matrix
    ft = np.empty((nstat, nf), dtype='c16')          # 
    
    newstart = stime
    tap = cosTaper(nsamp, p=0.22)  # 0.22 matches 0.2 of historical C bbfk.c
    offset = 0
    relpow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    RPOW = np.empty((grdpts_x, grdpts_y), dtype='f8')
    
    
    while eotr:
        
        
        BAZ=[]
        SLOW=[]
        RPOW=[]
        APOW=[]
        for l,r in zip(f_octaves_left, f_octaves_right):
                    
            
            try:
                for i, tr in enumerate(stream):
                    dat = tr.data[spoint[i] + offset:
                                  spoint[i] + offset + nsamp]
                    dat = (dat - dat.mean()) * tap
                    ft[i, :] = np.fft.rfft(dat, nfft)[l:r]
            except IndexError:
                break
            
            ft = np.require(ft, 'c16', ['C_CONTIGUOUS'])
            relpow_map.fill(0.)
            abspow_map.fill(0.)
            RPOW.fill(0.)
            # computing the covariances of the signal at different receivers
            dpow = 0.
            for i in xrange(nstat):
                for j in xrange(i, nstat):
                    R[:, i, j] = ft[i, :] * ft[j, :].conj()
    #                if method == 1:
    #                    R[:, i, j] /= np.abs(R[:, i, j].sum()**2)   # why?
                    if i != j:
                        R[:, j, i] = R[:, i, j].conjugate()
                    else:
                        dpow +=  np.abs(R[:, i, j].sum())
            dpow *= nstat
            
            # Equation for conv beamforming
            # P(f) = e.H R(f) e
            # P(f) = steer^H dot R dot steer
            # where e is the covariance (steering) matrix and 
            # R is the cross-spectral matrix.        
            
            # Equation for Capons beamformer
            # P(f) = 1/(e.H R(f)^-1 e)
            # P(f) = 1 / (steer^H dot R dot steer)
            # R(f)^-1 can be nonsingular (i.e. det=0) so diagonal loading is 
            # applied (Capon1969).
            if diag == True:
    
                # translated from gal code
                I = np.identity(nstat)
                for n in xrange(nf):
    
                    # calculate weights (capon1969)
                    # Vectorised version of nice DOA codes by M. Gal available 
                    # at https://github.com/mgalcode
                    weights = I*R[n, :, :].real.trace()/(nsamp)*diag_fact
                    R[n, :, :].real += weights
                    w = np.abs(R[n].diagonal())**-0.5  
                    wmean = np.sum(1./w**2)
                    wmean /= float(nstat)*(nhigh)                    
                    w_prod = np.outer(w, w.T)
                    R[n] *= w_prod
      
            for n in xrange(nf):
                R[n, :, :] = np.linalg.pinv(R[n, :, :], rcond=1e-6)
                
                
    #        # loop over fbins
    #        for n in xrange(nf):            
    #            # calculate capon spectrum - python version.
    #            freq_relpow_map = np.zeros(np.shape(relpow_map))
    #            for i in range(grdpts_x):
    #                for j in range(grdpts_y):
    #                    nf_steer = steer[n,i,j]
    #                    freq_relpow_map[i,j] = 1. / nf_steer.T.conj().dot(R[n,:,:]).dot(nf_steer)
    #            
    #            
    #            ix, iy = np.unravel_index(freq_relpow_map.argmax(), relpow_map.shape)
    #            relpow = freq_relpow_map[ix, iy]
    #            # here we compute baz, slow
    #            slow_x = sll_x + ix * sl_s
    #            slow_y = sll_y + iy * sl_s
    #    
    #            slow=np.sqrt(slow_x ** 2 + slow_y ** 2)
    #            if slow < 1e-8:
    #                slow = 1e-8
    #            fslow.append(slow)
    #            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
    #            fbaz.append(azimut % -360 + 180)        
                
            
            errcode = clibsignal.generalizedBeamformer(
                relpow_map, abspow_map, steer, R, nsamp, nstat, prewhiten,
                grdpts_x, grdpts_y, nfft, nf, dpow, method) 
            if errcode != 0:
                msg = 'generalizedBeamforming exited with error %d'
                raise Exception(msg % errcode)
                
                
            ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
            relpow, abspow = relpow_map[ix, iy], abspow_map[ix, iy]
            if store is not None:
                store(relpow_map, abspow_map, offset)
            # here we compute baz, slow
            slow_x = sll_x + ix * sl_s
            slow_y = sll_y + iy * sl_s
    
            slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
            if slow < 1e-8:
                slow = 1e-8
            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
            baz = azimut % -360 + 180
    
            BAZ.append(baz)
            SLOW.append(slow)
            APOW.append(abspow)
            RPOW.append(relpow)
            
            
            weight_by_rpower = False
            # Histogram the data
            abins = np.linspace(0, 2*np.pi, 72)   # 0 to 360 in steps of 360/N.
            sbins = np.linspace(0., 0.5, 20) 
            fbins = np.linspace(0., 0.25, 10)
            fbaz = np.asarray(fbaz)
            fbaz[fbaz<0] += 360
    
               
            f_hist, xedges, yedges = np.histogram2d(np.radians(np.asarray(fbaz)), 
                                                    f_octaves, bins=(abins, fbins))
                                                    
            f_hist_stack += f_hist                                       
                                                    
                                                    
    #        s_hist, xedges, yedges = np.histogram2d(np.radians(fbaz), fslow, 
    #                                               bins=(abins,sbins))                                                
    #                                                
    #                                                
    #        s_hist, xedges, yedges = np.histogram2d(np.radians(fbaz), f_octaves,
    #                                                bins=(abins, fbins))         
    #        
            
            
    #        if relpow > semb_thres and 1. / slow > vel_thres:
    #            res.append(np.array([newstart.timestamp, relpow, abspow, baz,
    #                                 slow]))
    #            if verbose:
    #                print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
            if (newstart + (nsamp + nstep) / fs) > etime:
                eotr = False
            offset += nstep
    
            newstart += nstep / fs
            
            
            
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
        # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719163
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
        

    return newstart.timestamp, f_hist_stack, xedges, yedges














#%%

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

def param_estimation(rpow, thresh, tstamp, sll_x, sll_y, sl_s):
    """
    Takes an array and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    
    
    :type rpow: numpy array
    :param rpow: input array of slowness space from array_processing
    :type thresh: float
    :param thresh: fraction of array maximum to use as cutoff (0-1).
    :type tstamp: UTCDateTime object
    :param tstamp: timestamp for subwindow
    :type sll_x: float
    :param sll_x: min slowness in x direction
    :type sll_y: float
    :param sll_y: min slowness value in y direction
    """


    # max value
    max_v = rpow.max()
    cutoff = max_v * thresh
    
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(rpow, footprint=neighborhood)==rpow

    # get indices
    inds = np.where((local_max==True) & (rpow>cutoff))
    
    # parameter estimation
    # find power vals for maxs
    relpow = rpow[inds]
            
    # convert indices to slowness components    
    ix, iy = np.asarray(inds)
    slow_x = sll_x + ix * sl_s
    slow_y = sll_y + iy * sl_s
    
    # convert components to slowness (magnitude)
    slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
    
    # convert components to az/baz
    az = 180. * np.arctan2(slow_x, slow_y) / np.pi    
    baz = az % -360 + 180
    baz[baz<0] += 360
    
    print '\n'
    print '--------------------------------------------------------------'
    print '------ Parameter Estimation: '+str(tstamp)+'------'
    print '--------------------------------------------------------------'
    print 
    print 'normalized power (dB)   ', 'velocity (km/s)   ', 'backazimuth (deg)'    
    for i in range(len(relpow)):
        print '%12.02f %19.02f %19.02f'%(relpow[i],slow[i]**-1,baz[i])


    return relpow, slow, baz



def plot_slow_space(rpow, net, tstamp, cfreq, flim,
                     sll_x, sll_y, slm_x, slm_y, sl_s,
                     plotbaz=False):

    rpow = 10*np.log10(rpow/rpow.max())
    
    #generating figure
    fig=plt.figure(figsize=(7,7), facecolor='white', 
                   edgecolor='lightsteelblue')
    ax=fig.add_subplot(1,1,1, aspect=1)
    slx = np.arange(sll_x-sl_s, slm_x, sl_s)
    sly = np.arange(sll_y-sl_s, slm_y, sl_s) 
    im = ax.pcolormesh(slx, sly, rpow, cmap='gist_stern_r')
    plt.title(net+': '+str(tstamp))# at %.03f +- %.03f[Hz]' %(cap_find/(nsamp*dt),cap_fave/(nsamp*dt)))
    ax.set_xlim([sll_x,slm_x])
    ax.set_ylim([sll_y,slm_y])
    ax.set_xlabel('East/West Slowness [s/km]')
    ax.set_ylabel('North/South Slowness [s/km]')
#    if plotbaz == True:
#        ax.invert_xaxis()
#        ax.invert_yaxis()
    ax.vlines(0, sll_y, slm_y, color='w', alpha=0.4)
    ax.hlines(0, sll_x, slm_x, color='w', alpha=0.4)
    ax.grid()
    circle=plt.Circle((0,0),sp.sqrt((0.3)**2),color='w',fill=False,alpha=0.4)
    plt.gcf().gca().add_artist(circle)
    circle=plt.Circle((0,0),sp.sqrt((0.24)**2),color='w',fill=False,alpha=0.4)
    plt.gcf().gca().add_artist(circle)
    cbar = fig.colorbar(im, shrink=0.7)
    cbar.set_label('relative power (dB)',rotation=270, labelpad=20)
               
               


def plotFK(out):
  
    """
    plots output from obspy.signal.array_processing
  
    :type out: numpy array
    :param out: output from obspy.signal.array_processing
    """
  

    # Plot
    labels = ['Normalised\nPower\n[dB]', 'Back-Az\n[$^\circ$]', 'Slowness\n[s/km]']
    
    fig = plt.figure(figsize=(10,5), facecolor='white', edgecolor='lightsteelblue')
 
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.scatter(out[:,0], out[:, i + 1], c=out[:, 1], alpha=1, marker = 'H',
                   edgecolors='none', cmap=plt.cm.gnuplot, s=30)
        ax.set_ylabel(lab, fontsize=16)
        ax.set_xlim(out[0, 0], out[-1, 0])
        print i
        if i < 2:
            ax.set_xticklabels([])
        if lab == 'Back-Az\n[$^\circ$]':
            ax.set_ylim(0, 360)
            
        elif lab == 'Normalised Power\n[dB]':
            ax.set_ylim(out[:, i + 1].min()*1.1, 0)
        elif lab == 'Slowness\n[s/km]':
            ax.set_ylim(0, out[:, i + 1].max()*1.1)
            
        
        ax.yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
#        else:
#            ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
#        
        ax.grid()



    l = ax.get_xticks()
    new_lab = [str(UTCDateTime(d))[:-8] for d in l]
    ax.set_xticklabels(new_lab, rotation=15)
    
    fig.subplots_adjust(left=0.18, top=0.95, right=0.95, bottom=0.2, hspace=0)
    plt.show()  
  
  
  
  
  
def polarFK(out, num=4):
    """
    plots output from obspy.signal.array_processing
  
    :type out: numpy array.
    :param out: output from obspy.signal.array_processing.
    :type num: int.
    :param num: number of ticks on radial axis.    
    
    """
  

    t, rel_power, baz, slow = out.T
    
    # scale markers by slowness (larger ourwards)    
    siz = 200*slow**2

    fig = plt.figure(figsize=(5,5), facecolor='white', 
                     edgecolor='lightsteelblue')
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta = np.radians(baz)
    c = ax.scatter(theta, slow, c=rel_power, s=siz, cmap=plt.cm.gnuplot)
    c.set_alpha(0.75)
    ax.yaxis.set_major_locator(MaxNLocator(num))
    plt.show()
  


def plot_polar_hist(out, r_pts=30, t_pts=72, weight_by_rpower=False):
    """
    Plot a polar histogram plot, with 0 degrees at the North.
     :type out: list
     :param out: output from array_processing
     :type r_pts: int or float
     :param r_pts: number of bins in radial direction, default=30.
     :type t_pts: int or float
     :param t_pts: number of bins in theta direction, default=72 i.e. 5 degree bins.
     
     Notes:
     params r_pts and t_pts can be made much small for large amount of data.
    """
    
    
    t, rel_power, baz, slow = out.T
    
    import matplotlib.pyplot as plt
    import numpy as np


    fig = plt.figure(figsize=(8,8), facecolor='white', 
                     edgecolor='lightsteelblue')
    ax = fig.add_subplot(111, polar=True)    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    
    # Histogram the data
    abins = np.linspace(0, 2*np.pi, t_pts)      # 0 to 360 in steps of 360/N.
    sbins = np.linspace(0, 0.5, r_pts) 
    
    if weight_by_rpower == True:
        H, xedges, yedges = np.histogram2d(np.radians(baz), slow, 
                                           bins=(abins,sbins), 
                                           weights=rel_power)
    else:
        H, xedges, yedges = np.histogram2d(np.radians(baz), slow, 
                                           bins=(abins,sbins))
        
    mH = np.ma.masked_where(H==0, H)
    #Grid to plot your data on using pcolormesh
    theta, r = np.mgrid[0:2*np.pi:t_pts*1j, 0:0.5:r_pts*1j]

    im = ax.pcolormesh(theta, r, mH, cmap=plt.cm.gist_stern_r)
    ax.grid()
    cbar = plt.colorbar(im, label='Normalised Power [dB]', 
                        shrink=0.7, pad=0.1)
                        
    if weight_by_rpower == True:
        cbar.ax.invert_yaxis() 
    
    plt.show()
    
    

    
