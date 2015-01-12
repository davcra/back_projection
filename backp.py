# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 20:38:23 2015

@author: davcra
"""

import scipy as sp
import numpy as np
import scipy.ndimage
from obspy.signal.array_analysis import *
from obspy.core import UTCDateTime, read, AttribDict
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap.pyproj as pyproj





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
    

def get_baz(lon1, lat1, lon2, lat2):


    g = pyproj.Geod(ellps='WGS84')
    az, baz, dist = g.inv(lon1, lat1, lon2, lat2)
    return az, baz, dist


def create_grid(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, step, clon, clat):

    """
    :clon  : longitude of array centre
    :clat  : latitude of array centre
    """
    
    lon_0 = (urcrnrlon + llcrnrlon) / 2.
    lat_0 = (urcrnrlat + llcrnrlat) / 2.    
    
    ny = int((urcrnrlat - llcrnrlat) / step)
    nx = int((urcrnrlon - llcrnrlon) / step)
  
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
   	           resolution='c',projection='lcc',
                lon_0=lon_0,lat_0=lat_0)    

    lons, lats = m.makegrid(nx, ny)     

    # compute map proj coordinates.
    xx, yy = m(lons, lats)
    
    # calculate baz along great circle from array centre to each grid point.
    bazgrid = []
    for lon1, lat1 in zip(np.concatenate(lons), np.concatenate(lats)):
        az, baz, dist = get_baz(lon1, lat1, clon, clat)
        bazgrid.append(baz)
    bazgrid = np.reshape(np.asarray(bazgrid), np.shape(xx))
    bazgrid[bazgrid < 0] += 360
    
    return lons, lats, bazgrid



def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]
    

def f_stat(S, M):
    
    F = (S/(1-S))*(M-1)

    return F


def dump(pow_map, apow_map, fstat_map, i):
    """
    Example function to use with `store` kwarg in
    :func:`~obspy.signal.array_analysis.array_processing`.
    """
    np.savez('/home/davcra/DUMP/pow_map_%d.npz' % i, pow_map)
    np.savez('/home/davcra/DUMP/apow_map_%d.npz' % i, apow_map)
    np.savez('/home/davcra/DUMP/fstat_map_%d.npz' % i, fstat_map)

def back_project_semblance(stream, interp, win_len, win_frac, 
                           sll_x, slm_x, sll_y, slm_y, sl_s, semb_thres, 
                           vel_thres, frqlow, frqhigh, stime, etime, prewhiten, 
                           verbose=False, coordsys='lonlat', timestamp='mlabday', 
                           method=0, store=None):
    
    """
    Method for Seismic-Array-Beamforming/FK-Analysis/Capon
    Note capon does not work (possible singular matrix).
    
    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type interp: string
    :param interp: method of interpolation to use when getting semblance values 
        for specific slowness, 'cubic' for cubic spline or 'nearest' for nearest 
        neighbour.
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
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness
    """
    
    _BF, CAPON = 0, 1
    res = []
    semblance = []
    num = 500
    vbaz = np.degrees(np.linspace(0,2*np.pi,num))
    names = []
    for tr in stream:
        names.append(tr.stats.station)
    names = list(set(names))
    M = len(names)    
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    
    # number of points in x and y directions
    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)
    
    # get lat lon
    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    # create time shift table
    time_shift_table = get_timeshift(geometry, sll_x, sll_y,
                                     sl_s, grdpts_x, grdpts_y)
        

    # start and end offsets relative to stime and etime for each
    # trace in stream in samples
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
    nhigh = min(nfft / 2 - 1, nhigh)  # avoid using nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency
    
    # to spead up the routine a bit we estimate all steering vectors in advance
    steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16')
    clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                         deltaf, time_shift_table, steer)
                         
    R = np.empty((nf, nstat, nstat), dtype='c16')
    ft = np.empty((nstat, nf), dtype='c16')
    newstart = stime
    tap = cosTaper(nsamp, p=0.22)  # 0.22 matches 0.2 of historical C bbfk.c
    offset = 0
    relpow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
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
        # computing the covariances of the signal at different receivers
        dpow = 0.
        for i in xrange(nstat):
            for j in xrange(i, nstat):
                R[:, i, j] = ft[i, :] * ft[j, :].conj()
                if method == CAPON:
                    R[:, i, j] /= np.abs(R[:, i, j].sum())
                if i != j:
                    R[:, j, i] = R[:, i, j].conjugate()
                else:
                    dpow += np.abs(R[:, i, j].sum())
        dpow *= nstat
        if method == CAPON:
            # P(f) = 1/(e.H R(f)^-1 e)
            for n in xrange(nf):
                R[n, :, :] = np.linalg.pinv(R[n, :, :], rcond=1e-6)

        errcode = clibsignal.generalizedBeamformer(
            relpow_map, abspow_map, steer, R, nsamp, nstat, prewhiten,
            grdpts_x, grdpts_y, nfft, nf, dpow, method)
        if errcode != 0:
            msg = 'generalizedBeamforming exited with error %d'
            raise Exception(msg % errcode)
	


 
        ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
        relpow, abspow = relpow_map[ix, iy], abspow_map[ix, iy]
        
        # convert semblance to f-statistic (Douze & Laster, 1979)
        fstat_map = f_stat(relpow_map, M)        
        
        if store is not None:
            store(relpow_map, abspow_map, fstat_map, offset)
        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180
        if baz < 0: baz += 360
            
            
            
        #######################################################################
	   # Here we extract semblance values at each baz for a particular slowness.
        
        # Interpolate the slowness at "num" points...
        X, Y = np.mgrid[sll_x:slm_x:sl_s, sll_y:slm_y:sl_s]

        # Define circle (radius is slowness)
        x = 0
        y = 0
        r = slow
        ang=np.linspace(-np.pi, np.pi, num)
        xp=r*np.cos(ang)
        yp=r*np.sin(ang)
        x_world = x + xp
        y_world = y + yp

        # baz samples
        az = np.linspace(-180, 180, num)
        az = az[:-1]
        vbaz = az % -360 + 180
        vbaz[vbaz<0.0] += 360
      
        col = relpow_map.shape[1] * (x_world - X.min()) / X.ptp()
        row = relpow_map.shape[0] * (y_world - Y.min()) / Y.ptp()
      
        # Extract the values along the circumfrence, using cubic interpolation
        if interp == 'cubic':
            interp_semb = scipy.ndimage.map_coordinates(fstat_map, 
                                                      np.vstack((row, col)))
	   # Or for nearest neighbour
        elif interp == 'nearest':
            interp_semb = relpow_map[row.astype(int), col.astype(int)] 
        
        interp_semb = np.asarray(interp_semb[:-1])
        semblance.append([newstart.timestamp, vbaz, interp_semb])
	   
	   
	   ######################################################################     
            
            
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
        res[:, 0] = res[:, 0] / (24. * 3600) + 719162
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res), semblance
    
    

def back_project_mult_array(data_dirs, coordfiles, t1, t2):
    """
    Run array processing for multiple arrays. 
    Need to add functionality to provide kwargs to be read in for each array.
    :type data_dirs: list
    :param data_dirs: list of directories each one pointing to data for a 
        specific array.
    :type coordfiles: list
    :param coordfiles: list of paths to coordinate files for insert_coordinates
    :type t1: UTCDateTime object
    :param t1: starttime
    :type t2: UTCDateTime object
    :param t2: endtime
    :type :
    :param :    
    """
    
    OUT = []
    SEMB = []
    GEOM = []    
    
    # loop through arrays
    for data_dir, coordfile in zip(data_dirs, coordfiles):        
        
        st = read(data_dir, starttime=t1, endtime=t2)
        st.merge()
        
        # ensure evan number of points in traces
        for tr in st:
            if tr.stats.npts & 1:
                tr.data = tr.data[:-1]
        
        # put coords inheaders
        st = insert_coordinates(st, coordfile)
    
        # run array processing
        out, semb = back_project_semblance(stream=st, interp='cubic', 
                                                win_len=600, win_frac=0.5, 
                                                sll_x=-0.5, slm_x=0.5, sll_y=-0.5, 
                                                slm_y=0.5, sl_s=0.03, 
                                                semb_thres=1e-9, vel_thres=-1e-9, 
                                                frqlow=0.1, frqhigh=0.2, stime=t1, 
                                                etime=t2-1, prewhiten=False, 
                                                verbose=False, coordsys='lonlat', 
                                                timestamp='julsec', method=0, 
                                                store=dump)
        # get geometry
        geometry = get_geometry(st, coordsys='lonlat', return_center=True)  
        
        OUT.append(out)
        SEMB.append(semb)
        GEOM.append(geometry)
   
    return OUT, SEMB, GEOM
    

def back_project_map(OUT, SEMB, GEOM, 
                     llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, step,
                     plot=True):   
   
   
   
    # define map center
    lon_0 = (urcrnrlon + llcrnrlon) / 2.
    lat_0 = (urcrnrlat + llcrnrlat) / 2.   
    
    # num points in x and y dirs
    ny = int((urcrnrlat - llcrnrlat) / step)
    nx = int((urcrnrlon - llcrnrlon) / step)

    # some variables
    fullgrid = np.ones((ny, nx))
    num = len(OUT[0])
    
    # loop through times
    for i in xrange(num):
        
        # create BAZ grid for array 1
        t, baz1, s1 = SEMB[0][i]
        clon1, clat1, celv1 = GEOM[0][-1]
        lons, lats, bazgrid1 =  create_grid(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, 
                               step, clon1, clat1)

        x1, y1 = np.shape(bazgrid1)
        sembgrid1 = np.zeros((x1, y1))        
        
        # create semb grid for array 1
        for p in range(x1):
            for q in range(y1):
                b = bazgrid1[p,q]
                idx, val = find_nearest(baz1, b)  # could do this better!
                sembgrid1[p,q] = s1[idx] 
                    
        # create BAZ grid for array 2
        t, baz2, s2 = SEMB[1][i]
        clon2, clat2, celv2 = GEOM[1][-1]
        lons, lats, bazgrid2 =  create_grid(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, 
                               step, clon2, clat2)

        x2, y2 = np.shape(bazgrid2)
        sembgrid2 = np.zeros((x2, y2))
                
        # create semb grid for array 2     
        for p in range(x2):
            for q in range(y2):
                b = bazgrid2[p,q]
                idx, val = find_nearest(baz2, b)
                sembgrid2[p,q] = s2[idx] 
        
        sembgrid = sembgrid1 * sembgrid2
        fullgrid *= sembgrid
        

    # plot map        
    if plot:
        plt.figure()
        m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                    urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                    resolution='c',projection='lcc',
                    lon_0=lon_0,lat_0=lat_0)
        m.drawcoastlines()
        
        xx, yy = m(lons, lats)
        m.pcolormesh(xx, yy, fullgrid)
        plt.colorbar()  
        
        
        inds = np.where(fullgrid==fullgrid.max())       
        slon = lons[inds]
        slat = lats[inds]
        x,y = m(slon,slat)
        plt.plot(x, y, color='k', marker='+', markersize=10)        
        plt.title(str(UTCDateTime(t1)))
        plt.savefig('/home/davcra/Desktop/test_figs/'+str(t1)+'.png')
        plt.close()
    return fullgrid




#%%


# how to run
stime = UTCDateTime(2013,5,1)
etime = UTCDateTime(2013,6,1)
starttimes = np.arange(np.float(stime), np.float(etime), 3600)
starttimes = [UTCDateTime(t) for t in starttimes]

for t1 in starttimes:
    
    t2 = t1+3600
    day = str(t1.julday).zfill(3)
    data_dirs = ['/media/davcra/SAMSUNG/CORRECTED/SCOTLAND/*/BHZ.C/*2013.'+day+'*',
                 '/media/davcra/SAMSUNG/CORRECTED/DONEGAL/*/HHZ.C/*2013.'+day+'*']
        
    coordfiles = ['/home/davcra/ARRAY_PROCESSING/EKA.txt',
                  '/home/davcra/ARRAY_PROCESSING/DONEGAL_COORDS.txt']
    llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = (-30.0, 40.0, 10.0, 65.0)
    step = 0.2

    OUT, SEMB, GEOM = back_project_mult_array(data_dirs, coordfiles, t1, t2)
    fullgrid = back_project_map(OUT, SEMB, GEOM, 
                               llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, step)
                     
                     
