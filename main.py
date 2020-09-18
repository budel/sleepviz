import os, csv
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw  
import numpy as np

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


def loadCSV(file):
    sleep = []
    
    with open(file, 'r') as csvfile:
        myreader = csv.reader(csvfile)
        row = next(myreader)  # gets the first line
        if row[0] != 'sid' or row[1] != 'start' or row[2] != 'stop' or row[3] != 'rating':
             raise Exception("Wrong csv header")

    with open(file, 'r') as csvfile:
        myreader = csv.reader(csvfile)
        for i, row in enumerate(myreader):
            sleep.append({})
            sleep[i]['sid'] = row[0]
            sleep[i]['start'] = row[1]
            sleep[i]['stop'] = row[2]
            sleep[i]['rating'] = row[3]
    return sleep

@app.route('/uploads/<filename>')
def uploaded_file(filename):

    basefile = os.path.join(app.config['UPLOAD_FOLDER'], 'base.csv')
    sleep1 = loadCSV(basefile)
    sleep2 = loadCSV(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    for s2 in sleep2:
        if s2['start'] in [s1['start'] for s1 in sleep1] and s2['stop'] in [s1['stop'] for s1 in sleep1]:
            continue
        #TODO add overlapping test
        sleep1.append(s2)
    
    sleep1 = sorted(sleep1, key=lambda x: x['start'])
    sleep1[1:], sleep1[0] = sleep1[:-1], sleep1[-1]  # put header back to front
    
    with open(basefile, 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        for s1 in sleep1:
            mywriter.writerow([s1['sid'], s1['start'], s1['stop'], s1['rating']])

    return redirect(url_for('render_image'))


def fill(pixels, s, firstdate, datewidth, color=(127,127,127)):
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

@app.route('/image')
def render_image():

    basefile = os.path.join(app.config['UPLOAD_FOLDER'], 'base.csv')
    sleep = []
    firstdate = datetime.max
    lastdate = datetime.min
    with open(basefile, 'r') as csvfile:
        myreader = csv.reader(csvfile)
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

    datewidth=900//numdates
    img = Image.new( 'RGB', (numdates*datewidth,24*60), "white")
    pixels = img.load()
    for s in sleep:
        fill(pixels, s, firstdate, datewidth)

    # create average day column
    imgarr = np.asarray(img)
    mean_img = linearMovingAverage(imgarr[:, ::datewidth].astype(float))
    saveMeanImg(mean_img, 'static/weightedmeanbabysleep.png', 50)
    #mean_img = imgarr.mean(axis=1)
    #saveMeanImg(mean_img, 'static/meanbabysleep.png', 20)

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r'static/open-sans.ttf', 16)  
    for j in range(0, 24*60, 60):
        for i in range(img.size[0]):
            pixels[i, j] = (0, 0, 0)
        draw.text((10, j), str(j//60), font=font, fill=(0, 0, 0))  
    
    img.save('static/babysleep.png', 'png')

    return '''
    <img src="static/babysleep.png" alt="babysleep">
    <img src="static/weightedmeanbabysleep.png" alt="weightedmeanbabysleep">
    <!--
    <img src="static/meanbabysleep.png" alt="meanbabysleep">
    -->
    '''

def linearMovingAverage(imgarr):
    n = imgarr.shape[1]
    mean_img = [(i+1) * imgarr[:, i] for i in range(n)]
    return 2/(n*(n+1)) * sum(mean_img)

def saveMeanImg(mean_img, out_file, sz=50):
    mean_img = ((mean_img - mean_img.min()) * (1/(mean_img.max() - mean_img.min()) * 255))
    mean_img = np.expand_dims(mean_img, axis=1)
    mean_img = np.tile(mean_img, (1,sz,1))
    mean_img = Image.fromarray(np.uint8(mean_img))
    mean_img.save(out_file, 'png')


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    # response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    # response.headers['Pragma'] = 'no-cache'
    # response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')
