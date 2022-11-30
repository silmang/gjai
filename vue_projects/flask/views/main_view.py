from flask import Blueprint, render_template

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def main_page():
    # return 'Hello, Pybo!'
    # return render_template("main/main_page.html")
    return render_template("main/main_page.vue")