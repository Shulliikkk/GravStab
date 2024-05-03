from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

def odeint(m, c, beta, alpha, l_0, phi_0):
    G = 6.6743e-11
    M = 5.972e24
    mu = G * M
    R = 6371000 + 400000
    Omega = (mu / R**3)**0.5
    Per = 2 * np.pi / Omega

    T = 250000
    x_0 = np.array([phi_0, 0, 2 * l_0, 0])
    t_0 = 0
    t_e = T
    t = np.linspace(0, T, 1000000)

    def right_part(t, x):
        right_part = np.zeros(4)
        right_part[0] = x[1]
        right_part[1] = - 2 * x[3] * x[1] / x[2] - mu / R**2 * alpha * (1 - alpha**2) / (alpha + 1) * np.sin(x[0]) / x[2] - mu / R**3 * np.sin(2 * x[0])
        right_part[2] = x[3]
        right_part[3] = x[1]**2 * x[2] + mu / R**2 * alpha * (1 - alpha**2) / (alpha + 1) * np.cos(x[0]) + 2 * mu / R**3 * x[2] * np.cos(x[0])**2 + mu / R**3 * x[2] - c / m * (x[2] + x[2] / alpha - l_0) + m * (alpha + 1) * mu / R**2 * np.cos(x[0]) - beta * x[3]
        return right_part

    x = solve_ivp(right_part, [t_0, t_e], x_0, t_eval = t, method = 'Radau')
    t = t / Per
    plt.plot(t, x.y[0] * 180 / np.pi)
    plt.xlabel(r'$t, per$')
    plt.ylabel(r'$\theta, grad$')
    plt.show()
    

class MyApp(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.initUI()

    def initUI(self):
        self.mylayout = QVBoxLayout()
        self.setLayout(self.mylayout)

        self.input_field_1 = QLineEdit()
        self.input_field_2 = QLineEdit()
        self.input_field_3 = QLineEdit()
        self.input_field_4 = QLineEdit()
        self.input_field_5 = QLineEdit()
        self.input_field_6 = QLineEdit()

        self.button = QPushButton("OK")
        self.label = QLabel("Wait please")
        self.label.hide()

        self.mylayout.addWidget(self.input_field_1)
        self.mylayout.addWidget(self.input_field_2)
        self.mylayout.addWidget(self.input_field_3)
        self.mylayout.addWidget(self.input_field_4)
        self.mylayout.addWidget(self.input_field_5)
        self.mylayout.addWidget(self.input_field_6)

        self.label_1 = QLabel("Input satellite mass:")
        self.label_2 = QLabel("Input spring stiffness:")
        self.label_3 = QLabel("Input friction coefficient")
        self.label_4 = QLabel("Input mass ratio:")
        self.label_5 = QLabel("Input length of an unstretched spring:")
        self.label_6 = QLabel("Input initial angular perturbation:")

        self.mylayout.addWidget(self.label_1)
        self.mylayout.addWidget(self.input_field_1)
        self.mylayout.addWidget(self.label_2)
        self.mylayout.addWidget(self.input_field_2)
        self.mylayout.addWidget(self.label_3)
        self.mylayout.addWidget(self.input_field_3)
        self.mylayout.addWidget(self.label_4)
        self.mylayout.addWidget(self.input_field_4)
        self.mylayout.addWidget(self.label_5)
        self.mylayout.addWidget(self.input_field_5)
        self.mylayout.addWidget(self.label_6)
        self.mylayout.addWidget(self.input_field_6)

        self.mylayout.addWidget(self.button)
        self.mylayout.addWidget(self.label)

        self.button.clicked.connect(self.on_button_click)

        self.setWindowTitle("GravStab")
        self.show()

    def on_button_click(self):
        m = float(self.input_field_1.text())
        c = float(self.input_field_1.text())
        beta = float(self.input_field_1.text())
        alpha = float(self.input_field_1.text())
        l_0 = float(self.input_field_1.text())
        phi_0 = float(self.input_field_1.text())

        self.input_field_1.hide()
        self.input_field_2.hide()
        self.input_field_3.hide()
        self.input_field_4.hide()
        self.input_field_5.hide()
        self.input_field_6.hide()
        self.label_1.hide()
        self.label_2.hide()
        self.label_3.hide()
        self.label_4.hide()
        self.label_5.hide()
        self.label_6.hide()

        self.button.hide()
        self.label.show() 

        self.app.processEvents()

        odeint(m, c, beta, alpha, l_0, phi_0)
        self.label.hide()
        self.data_label = QLabel(f"Results:")
        self.mylayout.addWidget(self.data_label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp(app)
    sys.exit(app.exec_())
