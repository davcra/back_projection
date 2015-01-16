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
import mpl_toolkits.basemap.pyproj as pyproj
import time
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator
from obspy.signal.util import az2baz2az




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
                     store=None, plot=None, plotbaz=False):
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

    R = np.empty((nf, nstat, nstat), dtype='c16')    # cov matrix
    ft = np.empty((nstat, nf), dtype='c16')          # 
    
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
#                    R[:, i, j] /= np.abs(R[:, i, j].sum()**2)
                if i != j:
                    R[:, j, i] = R[:, i, j].conjugate()
                else:
                    dpow +=  np.abs(R[:, i, j].sum())**2  # added **2
        dpow /= nstat
        

        
        
        if method == 1:
            # P(f) = 1/(e.H R(f)^-1 e)
            if diag == True:

                # diagonal loading as R(f)^-1 can be nonsingular (det=0)
                # translated from gal code
                I = np.identity(nstat)
                for n in xrange(nf):

                    # calculate weights
                    weights = I*R[n, :, :].real.trace()/(win_len)*diag_fact
                    R[n, :, :].real += weights
                    
                    
                    # applying weigths (check capon1969 for more info on the 
                    # equation)
                    
#                    # Gal version
#                    wmean = 0.0
#                    w = np.zeros(nstat)
#                    for i in range(nstat):
#                        w[i] = (R[n, :, :].real[i][i]*R[n, :, :].real[i][i]+\
#                                R[n, :, :].imag[i][i]*R[n, :, :].imag[i][i])**(-0.25)
#                        wmean += 1.0/(w[i]**2)
#                    
#                    for i in range(nstat):
#                        for j in range(nstat):
#                            R[n, :, :].real[i][j] *= w[i]*w[j]
#                            R[n, :, :].imag[i][j] *= w[i]*w[j]            
#                   
                    
                    # Vectorised version
                    w = np.abs(R[n].diagonal())**-0.5  
                    wmean = np.sum(1./w**2)
                    wmean /= float(nstat)*(nhigh)                    
                    w_prod = np.outer(w, w.T)
                    R[n] *= w_prod
  


            # P(f) = 1/(e.H R(f)^-1 e)
            for n in xrange(nf):
                R[n, :, :] = np.linalg.pinv(R[n, :, :], rcond=1e-6)
        
                
                # calculate capon spectrum
                for i in range(grdpts_x):
                    sx=-(sll_x+float(i*sl_s))
                    for j in range(grdpts_y):
                        sy=-(sll_y+float(i*sl_s))
                        nf_steer = steer[n,i,j]
                        relpow_map[i][j] = 1. / nf_steer.T.conj().dot(R[n,:,:]).dot(nf_steer)
                        
          
          
        
        #print steer, R, nsamp, nstat, prewhiten, nfft, nf, dpow, method
        else:
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
            net = st[0].stats.network
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
            res.append(np.array([newstart.timestamp, relpow, baz,
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

    im = ax.pcolormesh(theta, r, mH, cmap=colormap)
    ax.grid()
    cbar = plt.colorbar(im, label='Normalised Power [dB]', 
                        shrink=0.7, pad=0.1)
                        
    if weight_by_rpower == True:
        cbar.ax.invert_yaxis() 
    
    plt.show()
    
    

    
#%%


# how to run
T = []
R = []
B = []
S = []
RP = []
stime = UTCDateTime(2013,5,1)
#etime = UTCDateTime(2013,5,1, 12) #- 1800
etime = stime + 3600
while etime <= UTCDateTime(2013,5,1,1):




    day = str(stime.julday).zfill(3)
    
    
    data_dirs = ['/media/davcra/SAMSUNG/CORRECTED/SCOTLAND/*/BHZ.C/*2013.'+day+'*',
                 '/media/davcra/SAMSUNG/CORRECTED/DONEGAL/*/HHZ.C/*2013.'+day+'*']
            
    coordfiles = ['/home/davcra/ARRAY_PROCESSING/EKA.txt',
                  '/home/davcra/ARRAY_PROCESSING/DONEGAL_COORDS.txt']
    
    data_dir = data_dirs[1]
    coordfile = coordfiles[1]
    
    st = read(data_dir)
    st.merge()
    st = insert_coordinates(st, coordfile)
    
    
    # Execute array_processing
    kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-0.5, slm_x=0.5, sll_y=-0.5, slm_y=0.5, sl_s=0.01,
            # sliding window properties
            win_len=3600.0, win_frac=1,
            # frequency properties
            frqlow=0.125-0.0125, frqhigh=0.125+0.0125, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9, timestamp='julsec',
            stime=stime, etime=etime-1, diag=True, diag_fact=.01,
            method=1, plot=plot_slow_space)
        
    o, rpow = array_processing(st, **kwargs)
    t, relpow, baz, slow = o.T
    if baz < 0:baz += 360
    print 'Obspy baz: ',baz[0]
    print 'Obspy vel: ',1/slow[0]
    print 'Obspy rpow: ',relpow[0]
    RP.append(rpow)
    T.append(t)
    R.append(relpow)
    B.append(baz)
    S.append(slow)
    
    stime += 3600
    etime += 3600
#    
#T = np.concatenate(np.asarray(T))
#R = np.concatenate(np.asarray(R))
#B = np.concatenate(np.asarray(B))
#S = np.concatenate(np.asarray(S))
#out = np.array((T,R,B,S)).T
#plotFK(out)
#polarFK(out)
#plot_polar_hist(out)




#np.savetxt('/home/davcra/figs/out.txt', out, delimiter=" ")                  
#np.savetxt('/home/davcra/figs/rpow.txt', rpow, delimiter=" ")                  


#
#
##%%
#from obspy.core import read, UTCDateTime, AttribDict
#from obspy.signal import cornFreq2Paz
#
## Load data
#st = read("http://examples.obspy.org/agfa.mseed")
#
## Set PAZ and coordinates for all 5 channels
#st[0].stats.paz = AttribDict({
#    'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
#    'zeros': [0j, 0j],
#    'sensitivity': 205479446.68601453,
#    'gain': 1.0})
#st[0].stats.coordinates = AttribDict({
#    'latitude': 48.108589,
#    'elevation': 0.450000,
#    'longitude': 11.582967})
#
#st[1].stats.paz = AttribDict({
#    'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
#    'zeros': [0j, 0j],
#    'sensitivity': 205479446.68601453,
#    'gain': 1.0})
#st[1].stats.coordinates = AttribDict({
#    'latitude': 48.108192,
#    'elevation': 0.450000,
#    'longitude': 11.583120})
#
#st[2].stats.paz = AttribDict({
#    'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
#    'zeros': [0j, 0j],
#    'sensitivity': 250000000.0,
#    'gain': 1.0})
#st[2].stats.coordinates = AttribDict({
#    'latitude': 48.108692,
#    'elevation': 0.450000,
#    'longitude': 11.583414})
#
#st[3].stats.paz = AttribDict({
#    'poles': [(-4.39823 + 4.48709j), (-4.39823 - 4.48709j)],
#    'zeros': [0j, 0j],
#    'sensitivity': 222222228.10910088,
#    'gain': 1.0})
#st[3].stats.coordinates = AttribDict({
#    'latitude': 48.108456,
#    'elevation': 0.450000,
#    'longitude': 11.583049})
#
#st[4].stats.paz = AttribDict({
#    'poles': [(-4.39823 + 4.48709j), (-4.39823 - 4.48709j), (-2.105 + 0j)],
#    'zeros': [0j, 0j, 0j],
#    'sensitivity': 222222228.10910088,
#    'gain': 1.0})
#st[4].stats.coordinates = AttribDict({
#    'latitude': 48.108730,
#    'elevation': 0.450000,
#    'longitude': 11.583157})
#
#
## Instrument correction to 1Hz corner frequency
#paz1hz = cornFreq2Paz(1.0, damp=0.707)
#st.simulate(paz_remove='self', paz_simulate=paz1hz)
#
## Execute array_processing
#kwargs = dict(
#    # slowness grid: X min, X max, Y min, Y max, Slow Step
#    sll_x=.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
#    # sliding window properties
#    win_len=3.0, win_frac=0.05,
#    # frequency properties
#    frqlow=4.0, frqhigh=7.0, prewhiten=0,
#    # restrict output
#    semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
#    stime=UTCDateTime("20080217110515"), etime=UTCDateTime("20080217110545"),
#    method=1)
#
#o, rpow = array_processing(st, **kwargs)
#
#o.T[2][o.T[2]<0]+=360
#plotFK(o)
#
#
## Execute array_processing
#kwargs = dict(
#    # slowness grid: X min, X max, Y min, Y max, Slow Step
#    sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
#    # sliding window properties
#    win_len=1.0, win_frac=0.05,
#    # frequency properties
#    frqlow=4.0, frqhigh=7.0, prewhiten=0,
#    # restrict output
#    semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
#    stime=UTCDateTime("20080217110515"), etime=UTCDateTime("20080217110545"),
#    method=0)
#
#o2, rpow2 = array_processing(st, **kwargs)
#o2.T[2][o2.T[2]<0]+=360
#
#
#
#fig = plt.figure()
#ax1 = fig.add_subplot(211)
#ax1.scatter(o.T[0], o.T[2], c='k')
#ax1.scatter(o2.T[0], o2.T[2], c='r')
#ax1.set_xlim(o.T[0].min(), o.T[0].max())
#ax1.grid()
#
#ax2 = fig.add_subplot(212)
#ax2.scatter(o.T[0], o.T[3], c='k')
#ax2.scatter(o2.T[0], o2.T[3], c='r')
#ax2.set_xlim(o.T[0].min(), o.T[0].max())
#ax2.grid()
#
#
#
