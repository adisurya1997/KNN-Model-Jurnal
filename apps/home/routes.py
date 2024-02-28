# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os
from apps.home import blueprint
from flask import render_template, request, flash, redirect, url_for
from flask_login import login_required
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename
import csv
import pandas as pd
from apps.home.module.processed_data import build_csv_file as build
from apps.home.module.processed_data import processing as proces
from apps.home.module.processed_data import predicted
UPLOAD_FOLDER = 'apps/static/assets/filedata'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_best_accuracy(perf):
    best_accuracy = 0
    best_metrics = None
    for item in perf:
        if item["accuracy"] > best_accuracy:
            best_accuracy = item["accuracy"]
            best_metrics = item

    return best_metrics
starts = 0

@blueprint.route('/index')
@login_required
def index():
    df = pd.read_csv('apps/static/assets/filedata/paper.csv')
    category_counts = df["Primary Category"].value_counts().to_dict()
    categories = df["Primary Category"].unique().tolist()
    data_list = df.values.tolist()
    return render_template('home/index.html', segment='index',categories=categories,data_list=data_list,category_counts=category_counts)

@blueprint.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        # Pastikan file sudah diunggah
        if 'csv_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['csv_file']
        # Pastikan nama file tidak kosong
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # Pastikan file merupakan file CSV yang diizinkan
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER,filename))
            flash('File uploaded successfully')
            return render_template('home/index.html'), 200
        else:
            flash('Invalid file format. Please upload a CSV file.')
    return render_template('home/index.html')
@blueprint.route('/<template>', methods=['GET', 'POST'])
@login_required
def route_template(template):
    global starts
    global perf
    df = pd.read_csv('apps/static/assets/filedata/output.csv')
    try:
        if not template.endswith('.html'):
            template += '.html'
        elif template == "processed.html":
            segment = get_segment(request)
            success = build()
            data_list = df.values.tolist()
            return render_template("home/" + template, segment=segment,data_list=data_list)
        elif template == "confusion-matrix.html":
            segment = get_segment(request)

            # Serve the file (if exists) from app/templates/home/FILE.html
            return render_template("home/" + template, segment=segment)
        elif template == "tbl_performance.html":
            if request.method == 'POST':
                starts += 1
                input_n = request.form.get('input_n')
                n_metrick = request.form.get('metrick')
                input_n_list = [int(x) for x in input_n.split(',')]
                if n_metrick == 'all':
                    n_metrick = ["cosine","euclidean","manhattan"]
                    perf = proces("performa",input_n_list,n_metrick)
                    segment = get_segment(request)
                    best = get_best_accuracy(perf)
                    return render_template("home/" + template, segment=segment, perf = perf,best=best)
                n_metrick = [n_metrick]
                perf = proces("performa",input_n_list,n_metrick)
                segment = get_segment(request)
                best = get_best_accuracy(perf)
                return render_template("home/" + template, segment=segment, perf = perf,best=best)
            if starts == 0:
                starts += 1
                input_n = [17,19,21]
                n_metrick = ["cosine","euclidean"]
                perf = proces("performa",input_n,n_metrick)
                best = get_best_accuracy(perf)
                segment = get_segment(request)
                # Serve the file (if exists) from app/templates/home/FILE.html
                return render_template("home/" + template, segment=segment, perf = perf,best=best)
            print(starts)
            print(perf)
            perfs = perf
            best = get_best_accuracy(perfs)
            segment = get_segment(request)
            # Serve the file (if exists) from app/templates/home/FILE.html
            return render_template("home/" + template, segment=segment, perf = perf,best=best)
        elif template == "predict.html":
            if starts == 0:
                input_n = [17,19,21]
                n_metrick = ["cosine","euclidean"]
                perf = proces("performa",input_n,n_metrick)
                starts += 1
            if request.method == 'POST':
                text = request.form.get('text')
                text = str(text)
                perfs = predicted(text)
                segment = get_segment(request)
                return render_template("home/" + template, segment=segment, perfs = perfs,text=text)
            text = "kosong"
            segment = get_segment(request)
            perfs = predicted(text)
            return render_template("home/" + template, segment=segment, perfs = perfs,text=text)
        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except Exception as e:
        print("Error:", e)
        return render_template('home/page-500.html' ,error_message=str(e)), 500



# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
