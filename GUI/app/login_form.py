from PyQt6.QtWidgets import QDialog, QMessageBox
from ui_generated.login_ui import Ui_LoginForm
from app.db import get_connection
from passlib.hash import pbkdf2_sha256
from ui_generated.main_ui import VideoWindow
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPixmap
import os

class LoginForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginForm()
        self.ui.setupUi(self)
        # Set icon ở đây
        self.icon_top = os.path.join(os.path.dirname(__file__), "..", "Resources", "traffic.ico")
        self.setWindowIcon(QIcon(self.icon_top))
        # Đường dẫn đầy đủ đến icon avatar
        self.avatar_path = os.path.join(os.path.dirname(__file__), "..", "Resources", "icons8-user-50.png")
        pixmap = QPixmap(self.avatar_path)

        if pixmap.isNull():
            print("❌ Không load được ảnh avatar.")
        else:
            self.ui.ImagePicture.setPixmap(pixmap)
            self.ui.ImagePicture.setScaledContents(True)  # Để ảnh co theo QLabel

        self.ui.PasswordInput.setEchoMode(self.ui.PasswordInput.EchoMode.Password)
        self.ui.LoginButton.clicked.connect(self.handle_login)

        # Load settings
        self.settings = QSettings("MyCompany", "SecurityCameraApp")
        self.load_settings()

    def load_settings(self):
        saved_username = self.settings.value("username", "")
        remember = self.settings.value("remember", False, type=bool)

        self.ui.UsernameInput.setText(saved_username)
        self.ui.RememberMeCheckBox.setChecked(remember)
    
    def save_settings(self, username):
        if self.ui.RememberMeCheckBox.isChecked():
            self.settings.setValue("username", username)
            self.settings.setValue("remember", True)
        else:
            self.settings.remove("username")
            self.settings.setValue("remember", False)

    def handle_login(self):
        username = self.ui.UsernameInput.text().strip()
        password = self.ui.PasswordInput.text().strip()

        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT password, full_name FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()
            conn.close()

            if result:
                db_password, full_name = result
                db_password = db_password.strip()
                if pbkdf2_sha256.verify(password, db_password):
                    QMessageBox.information(self, "Login Success", f"Welcome, {full_name}!")
                    # self.accept()

                    # ✅ Save remember me state
                    self.save_settings(username)
                    
                    self.hide()
                    self.video_window = VideoWindow()
                    print("MO APP NE")
                    self.video_window.show()
                else:
                    QMessageBox.warning(self, "Login Failed", "Incorrect password.")
            else:
                QMessageBox.warning(self, "Login Failed", "User not found.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
