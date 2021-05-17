import csv
from datetime import datetime, timedelta
from PIL import Image, ImageFont, ImageDraw  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualizeCSV(csvfile):

    sleep, firstdate, lastdate = readCSV(csvfile)
    print_top(10, sleep)
    plot_duration_per_day(sleep)
    plot_phases_per_night(sleep)
    numdates = date_diff(firstdate, lastdate) + 1
    # Todo: Think about Dec/Jan
    plot_histogram(sleep, lastdate.month-1, lastdate.year)
    plot_histogram(sleep, lastdate.month, lastdate.year)

    offset_h = 7
    datewidth=950//numdates
    img = Image.new( 'RGB', (numdates*datewidth,24*60+25), "white")
    pixels = img.load()
    for s in sleep:
        fill_pixel(pixels, s, firstdate, datewidth, offset=offset_h)
    
    # create average day column
    imgarr = np.asarray(img)
    mean_img = linearMovingAverage(imgarr[:, ::datewidth].astype(float))
    mean_img = genMeanImg(mean_img, 2*datewidth)
    #mean_img = imgarr.mean(axis=1)
    #mean_img = genMeanImg(mean_img, 20)

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r'static/open-sans.ttf', 16)  
    for j in range(0, 24*60, 60):
        for i in range(img.size[0]):
            pixels[i, j] = (0, 0, 0)
        draw.text((img.size[0] - 20, j), str((j//60+offset_h)%24), font=font, fill=(0, 0, 0))  

    for i in range(0, numdates*datewidth, datewidth):
        datediff = i // datewidth
        c = firstdate + timedelta(datediff)
        weeksincebirth = (36+datediff)//7
        if c.weekday() == 0 and weeksincebirth%2:
            draw.text((i+datewidth, 24*60), str(weeksincebirth), font=font, fill=(0,0,0))
            for j in range(24*60,24*60+25):
                pixels[i, j] = (0, 0, 0)
        if c.day == 17:
            for j in range(24*60):
                pixels[i+datewidth//2, j] = (0, 0, 0)
            
    
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
            assert(sleep[i]['stop'] > sleep[i]['start'])
            sleep[i]['duration'] = sleep[i]['stop'] - sleep[i]['start']
            firstdate = firstdate if firstdate < sleep[i]['start'] else sleep[i]['start']
            lastdate = lastdate if lastdate > sleep[i]['stop'] else sleep[i]['stop']
    return sleep, firstdate, lastdate

def print_top(n, sleep):
    sleep_sorted = sorted(sleep, key=lambda x: x['duration'], reverse=True)
    for s in sleep_sorted[:n]:
        print(s['start'], s['duration'])

def plot_duration_per_day(sleep):
    cdates = [sleep[0]['start'].date()]
    cum_durations = [0]
    max_durations = [-np.inf]
    for s in sleep:
        sdate = s['start'].date()
        edate = s['stop'].date()
        dur = s['duration'] / timedelta(hours=1) 
        if cdates[-1] != sdate:
            cdates.append(edate)
            cum_durations.append(0)
            max_durations.append(-np.inf)
        max_durations[-1] = dur if dur > max_durations[-1] else max_durations[-1]
        if sdate == edate:
            cum_durations[-1] += dur
        else:
            rest_today = (datetime.combine(sdate + timedelta(days=1), datetime.min.time()) - s['start']) / timedelta(hours=1)
            cum_durations[-1] += rest_today
            cdates.append(edate)
            cum_durations.append(dur - rest_today)
            max_durations.append(-np.inf)
            
    fig, ax = plt.subplots()
    ax.bar(cdates, cum_durations, width=1.0)
    ax.bar(cdates, max_durations, width=1.0, color='red')
    ax.xaxis_date()
    ax.yaxis.grid(True)
    ax.autoscale(enable=True, axis='x', tight=True)
    fig.autofmt_xdate()
    plt.savefig('static/durations.png', dpi=160, bbox_inches='tight')

    
def plot_phases_per_night(sleep):
    start = datetime.strptime("19:00:00", "%H:%M:%S").time()
    end = datetime.strptime("07:00:00", "%H:%M:%S").time()
    cdates = [sleep[0]['start'].date()]
    counts = [0]
    for s in sleep:
        if is_between(s['start'].time(), start, end) and is_between(s['stop'].time(), start, end):
            counts[-1]+=1
        elif cdates[-1] != s['start'].date():
            counts.append(0)
            cdates.append(s['start'].date())
    fig, ax = plt.subplots()
    ax.bar(cdates, counts, width=1.0)
    ax.xaxis_date()
    ax.autoscale(enable=True, axis='x', tight=True)
    fig.autofmt_xdate()
    plt.savefig('static/phases.png', dpi=160, bbox_inches='tight')

def is_between(now, start, end):
    if (start < end):
        return start <= now and now <= end
    return not(end < now and now < start)


def plot_histogram(sleep, month, year, n_bins=288):
    fig, ax = plt.subplots()
    durations = [s['duration'] / timedelta(hours=1) for s in sleep if s['stop'].month == month and s['stop'].year == year]
    ax.hist(durations, bins=n_bins)
    ax.autoscale(enable=True, axis='x', tight=True)
    Ms = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.title(f'{Ms[month-1]} {year}')
    plt.savefig(f'static/histogram{year}_{month:02d}.png', dpi=160, bbox_inches='tight')


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
    # load dataset and parse timestamps
    df = pd.read_csv(base_csv)
    df[['start', 'stop']] = df[['start', 'stop']].apply(pd.to_datetime, unit='ms')
    df['start'] = df['start'].dt.tz_localize('utc').dt.tz_convert('Europe/Berlin')
    df['stop'] = df['stop'].dt.tz_localize('utc').dt.tz_convert('Europe/Berlin')

    # set origin at the first hour, correcting for utc
    first_trip = df['start'].min()
    origin = first_trip.replace(hour=0, minute=0, second=0)
    hours_ticks = np.arange(0, 24).tolist()

    ax = plt.subplot(111, projection='polar')
    for idx, event in df.iterrows():
        tstart, tstop = event.loc[['start', 'stop']]

        # convert trip timestamps to day fractions
        if tstart < pd.Timestamp(2020,10,25).tz_localize('Europe/Berlin'):
            tstart -= origin
            tstop -= origin
        else:
            tstart -= origin.replace(hour=1)
            tstop -= origin.replace(hour=1)
        tstart /= np.timedelta64(1, 'D')
        tstop /= np.timedelta64(1, 'D')

        # timestamps are in day fractions, 2pi is one day
        nsamples = int(1000. * (tstop - tstart))
        t = np.linspace(tstart, tstop, nsamples)
        theta = 2 * np.pi * t
        ax.plot(theta, t, lw=0.8, color='gray')

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
