# from flask import Flask
# def create_app(): # 약속된 함수
#     app = Flask(__name__)
    
#     # @app.route('/')
#     # def main_page():
#     #     return 'Hello, World!'
    
#     from views import main_view
#     app.register_blueprint(main_view.bp)
    
#     return app

from flask import Flask
app = Flask(__name__)

# @app.route("/")
# def main_page():
#     return 'Hello, World!'

from views import main_view
app.register_blueprint(main_view.bp)

if __name__ == "__main__":
    app.run(debug=True)