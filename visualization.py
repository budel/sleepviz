import csv
from datetime import datetime, timedelta
from PIL import Image, ImageFont, ImageDraw  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualizeCSV(csvfile):

    sleep, firstdate, lastdate = readCSV(csvfile)
    numdates = date_diff(firstdate, lastdate) + 1

    offset_h = 8
    datewidth=900//numdates
    img = Image.new( 'RGB', (numdates*datewidth,24*60), "white")
    pixels = img.load()
    for s in sleep:
        fill_pixel(pixels, s, firstdate, datewidth, offset=offset_h)
    
    # create average day column
    imgarr = np.asarray(img)
    mean_img = linearMovingAverage(imgarr[:, ::datewidth].astype(float))
    mean_img = genMeanImg(mean_img, 50)
    #mean_img = imgarr.mean(axis=1)
    #mean_img = genMeanImg(mean_img, 20)

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r'static/open-sans.ttf', 16)  
    for j in range(0, 24*60, 60):
        for i in range(img.size[0]):
            pixels[i, j] = (0, 0, 0)
        draw.text((10, j), str((j//60+offset_h)%24), font=font, fill=(0, 0, 0))  
    
    return img, mean_img


def readCSV(csvfile):
    sleep = []
    firstdate = datetime.max
    lastdate = datetime.min
    with open(csvfile, 'r') as f:
        myreader = csv.reader(f)
        next(myreader)
        for i, row in enumerate(myreader):
            sleep.append({})
            sleep[i]['sid'] = row[0]
            sleep[i]['start'] = datetime.fromtimestamp(int(row[1]) / 1000)
            sleep[i]['stop'] = datetime.fromtimestamp(int(row[2]) / 1000)
            sleep[i]['rating'] = row[3]
            firstdate = firstdate if firstdate < sleep[i]['start'] else sleep[i]['start']
            lastdate = lastdate if lastdate > sleep[i]['stop'] else sleep[i]['stop']
    return sleep, firstdate, lastdate


def date_diff(d1, d2):
    d1 = d1.replace(hour = 0)
    d1 = d1.replace(minute = 0)
    d1 = d1.replace(second = 0)
    d1 = d1.replace(microsecond = 0)
    d2 = d2.replace(hour = 0)
    d2 = d2.replace(minute = 0)
    d2 = d2.replace(second = 0)
    d2 = d2.replace(microsecond = 0)
    return (d2 - d1).days


def fill_pixel(pixels, s, firstdate, datewidth, color=(127,127,127), offset=0):
    sstart = s['start'] - timedelta(hours=offset)
    sstop = s['stop'] - timedelta(hours=offset)
    start = sstart.hour * 60 + sstart.minute
    stop = sstop.hour * 60 + sstop.minute
    diff = date_diff(firstdate, sstart)

    if sstart.day == sstop.day:
        for i in range(diff*datewidth, diff*datewidth+datewidth):
            for j in range(start, stop):
                pixels[i,j] = color
    else:
        for i in range(diff*datewidth, diff*datewidth+datewidth):
            for j in range(start, 24*60):
                pixels[i,j] = color
        for i in range((diff+1)*datewidth, (diff+1)*datewidth+datewidth):
            for j in range(0, stop):
                pixels[i,j] = color
       
def linearMovingAverage(imgarr):
    n = imgarr.shape[1]
    mean_img = [(i+1) * imgarr[:, i] for i in range(n)]
    return 2/(n*(n+1)) * sum(mean_img)

def genMeanImg(mean_img, sz=50):
    mean_img = ((mean_img - mean_img.min()) * (1/(mean_img.max() - mean_img.min()) * 255))
    mean_img = np.expand_dims(mean_img, axis=1)
    mean_img = np.tile(mean_img, (1,sz,1))
    mean_img = Image.fromarray(np.uint8(mean_img))
    return mean_img

# see https://stackoverflow.com/questions/46575723/creating-a-temporal-range-time-series-spiral-plot
def polarPlot(base_csv):
    UTC_TIMEDIFF = 2
    # load dataset and parse timestamps
    df = pd.read_csv(base_csv)
    df[['start', 'stop']] = df[['start', 'stop']].apply(pd.to_datetime, unit='ms')

    # set origin at the first hour, correcting for utc
    first_trip = df['start'].min()
    origin = (first_trip - pd.to_timedelta(first_trip.hour + UTC_TIMEDIFF, unit='h')).replace(minute=0, second=0)
    hours_ticks = np.arange(0, 24).tolist()

    # convert trip timestamps to day fractions
    df['start'] = (df['start'] - origin) / np.timedelta64(1, 'D')
    df['stop']  = (df['stop']  - origin) / np.timedelta64(1, 'D')

    ax = plt.subplot(111, projection='polar')
    for idx, event in df.iterrows():
        tstart, tstop = event.loc[['start', 'stop']]
        # timestamps are in day fractions, 2pi is one day
        nsamples = int(1000. * (tstop - tstart))
        t = np.linspace(tstart, tstop, nsamples)
        theta = 2 * np.pi * t
        ax.plot(theta, t, lw=3, color='gray')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(-5)
    ax.set_rticks([])  # No radial ticks
    ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
    ax.set_xticklabels(hours_ticks)

    plt.savefig('static/polarsleep.png', dpi=300, bbox_inches='tight')


def fill_array(imgarr, s, firstdate, offset=0):    
    sstart = s['start'] - timedelta(hours=offset)
    sstop = s['stop'] - timedelta(hours=offset)
    start = sstart.hour * 60 + sstart.minute
    stop = sstop.hour * 60 + sstop.minute
    diff = date_diff(firstdate, sstart)

    if sstart.day == sstop.day:
        for i in range(diff, diff+1):
            for j in range(start, stop):
                imgarr[i,j] = 0
    else:
        for i in range(diff, diff+1):
            for j in range(start, 24*60):
                imgarr[i,j] = 0
        for i in range(diff+1, diff+2):
            for j in range(0, stop):
                imgarr[i,j] = 0
       
    return imgarr

if __name__ == '__main__':
    img, img_mean = visualizeCSV('static/base.csv')
    # img.save('static/babysleep.png', 'png')
    # img_mean.save('static/weightedmeanbabysleep.png', 'png')
