import csv
from datetime import datetime, timedelta
from PIL import Image, ImageFont, ImageDraw  
import numpy as np

def visualizeCSV(csvfile):

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
    
    numdates = date_diff(firstdate, lastdate) + 1

    offset_h = 8
    datewidth=900//numdates
    img = Image.new( 'RGB', (numdates*datewidth,24*60), "white")
    pixels = img.load()
    for s in sleep:
        fill(pixels, s, firstdate, datewidth, offset=offset_h)

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


def fill(pixels, s, firstdate, datewidth, color=(127,127,127), offset=0):
    s['start'] = s['start'] - timedelta(hours=offset)
    s['stop'] = s['stop'] - timedelta(hours=offset)
    start = s['start'].hour * 60 + s['start'].minute
    stop = s['stop'].hour * 60 + s['stop'].minute
    diff = date_diff(firstdate, s['start'])

    if s['start'].day == s['stop'].day:
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


if __name__ == '__main__':
    img, img_mean = visualizeCSV('static/base.csv')
    img.save('static/babysleep.png', 'png')
    img_mean.save('static/weightedmeanbabysleep.png', 'png')