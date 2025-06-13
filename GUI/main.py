import os
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
from PyQt6.QtWidgets import QApplication
from app.login_form import LoginForm
import sys

def main():
    app = QApplication(sys.argv)
    login = LoginForm()
    # login.exec()
    login.show() 
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
