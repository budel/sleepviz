import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from curation import mergeCSV
from visualization import visualizeCSV, polarPlot
import threading

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):

    basefile = os.path.join(app.config['UPLOAD_FOLDER'], 'base.csv')
    newfile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mergeCSV(basefile, newfile)

    # thread = threading.Thread(target=polarPlot, args=(basefile, ))
    # thread.start()

    return redirect(url_for('render_image'))

@app.route('/image')
def render_image():

    basefile = os.path.join(app.config['UPLOAD_FOLDER'], 'base.csv')
    img, img_mean = visualizeCSV(basefile)
    img.save('static/babysleep.png', 'png')
    img_mean.save('static/weightedmeanbabysleep.png', 'png')

    return '''
    <img src="static/babysleep.png" alt="babysleep">
    <img src="static/weightedmeanbabysleep.png" alt="weightedmeanbabysleep">
    <img src="static/durations.png" alt="durations">
    <img src="static/phases.png" alt="phases">
    <!--
    <img src="static/polarsleep.png" alt="polarsleep">
    <img src="static/meanbabysleep.png" alt="meanbabysleep">
    -->
    '''


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
